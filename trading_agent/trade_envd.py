import gym
from numpy import exp, argmax, random

import numpy as np

#import plot

maker_fee = 0.01 * 0.01
taker_fee = 0.06 * 0.01
MAX_LEVERAGE = 25
fee_rate = 0.02 * 0.01
max_total_loss = 0.05
min_e = 0.2 * 0.01
LEVERAGE = 2

window = 600 #200
t0 = 600 #200
tick_size = 0.0001
sample_size = 1200 #800

number_of_episodes = 0
sum_scores = 0
visualize = False

class TradeEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        #self.action_space = np.array(['buy', 'sell', 'close_long', 'close_short'])
        self.symbol = 'ADA'
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(window, data.shape[1]))
        self.time = t0
        self.trading = False
        self.profit = 0
        self.order_list = []
        self.data = data
        n = random.randint(200, len(data) - sample_size)
        self.sample = data[n: n + sample_size].copy()
        self.state = self.sample[1 - window + self.time:self.time + 1]
        self.state_size = window * data.shape[1]
        self.score = 0
        self.num_trades = 0
        self.profit_t = 0
        self.profit_t1 = 0
        self.terminate = False

    def reset(self):
        #self.action_space = np.array(['buy', 'sell', 'close_long', 'close_short', 'd_target', 'd_stop_loss'])
        # self.action_space = np.array(['hold', 'buy', 'sell'])
        self.time = t0
        self.trading = False
        self.profit = 0
        self.order_list = []
        n = random.randint(200, len(self.data) - sample_size)
        self.sample = self.data[n: n + sample_size].copy()
        self.state = self.sample[1 - window + self.time:self.time + 1]
        self.score = 0
        self.num_trades = 0
        self.profit_t = 0
        self.profit_t1 = 0
        self.terminate = False

        return self.state

    def step(self, action):
        global number_of_episodes
        global sum_scores
        # [hold buy sell close_long close_short]
        close = self.sample[self.time, 0]
        ma = self.sample[self.time, 3]
        side = 0
        if action == 1:
            side = 1
        elif action == 2:
            side = -1

        e_target, e_stop_loss = 10, max_total_loss

        if not self.trading:
            if not self.terminate and side != 0 and self.check_params(side, e_target, e_stop_loss, self.time):
                self.trading = True
                self.num_trades += 1
            reward = 0
        else:
            order = self.order_list[-1]
            order.exit_price = self.sample[self.time][0]
            if (order.side == 1 and action == 3) or (order.side == -1 and action == 4):
                order = self.close_order(order)
            else:
                order = self.check_limits(self.time, order)

            self.profit_t1 = self.profit_t
            self.profit_t = order.calculate_profit()
            reward = np.log(1 + self.profit_t) - np.log(1 + self.profit_t1)
            self.score += reward
            if exp(self.score) < 0.7:
                self.terminate = True
                reward -= 0.8

        self.time += 1
        if self.trading:
            order = self.order_list[-1]
            # (close - entry) / (entry - stop_loss)
            self.sample[self.time, -1] = \
                (self.sample[self.time, 0] - order.entry_price)/(order.entry_price - order.stop_loss)
        self.state = self.sample[1 - window + self.time:self.time + 1]

        done = False
        if self.time >= sample_size - 1:
            if self.trading:
                order = self.order_list[-1]
                order.exit_price = self.sample[self.time][0]
                order = self.close_order(order)
                #self.profit_t1 = self.profit_t
                #self.profit_t = order.calculate_profit()
                #reward = np.log(1 + self.profit_t) - np.log(1 + self.profit_t1)
            #self.show_orders()
            number_of_episodes += 1
            sum_scores += self.score
            print(f"number of trades: {len(self.order_list)}, score: {exp(self.score) - 1}"
                  f", profit: {calculate_total_profit(self.order_list)}")
            if len(self.order_list) <= 1:
                reward -= 0.5
            done = True
            if visualize:
                self.visualize()
        info = {}
        #print(self.state)
        #reward = np.log(max(1 + reward, 0.1))
        #reward -= 10 * self.num_trades / self.time
        #print(self.state.shape)
        return self.state, reward, done, info


    def close_order(self, order):
        close = self.sample[self.time][0]
        order.exit_price = close
        self.trading = False
        order.calculate_profit()
        order.duration = self.time - order.start_time
        order.end_time = self.time
        #print('profit:', order.profit)
        return order


    def check_limits(self, time, order):
        close = self.sample[time][0]
        high = self.sample[time][1]
        low = self.sample[time][2]
        entry = order.entry_price

        stop_loss = order.stop_loss
        target = order.target
        side = order.side

        if side == 1:
            rr = (high - entry)/(entry - stop_loss)
        else:
            rr = (low - entry)/(entry - stop_loss)

        if rr > order.max_rr:
            order.max_rr = rr

        if side * (close - stop_loss) <= 0:
            order.exit_price = order.stop_loss
            self.trading = False
            order.calculate_profit()
            order.duration = time - order.start_time
            order.end_time = self.time
            #print('profit:',order.profit)
        elif side * (close - target) >= 0:
            i = order.targets.index(target)
            #if i == len(order.targets) - 1:
            order.exit_price = order.target
            self.trading = False
            order.calculate_profit()
            order.duration = time - order.start_time
            order.end_time = self.time
            #print('profit:', order.profit)
            return order

            """
            else:
                order.target = order.targets[i + 1]
                order.stop_loss = order.stop_losses[i + 1]
            """

        return order


    def close(self):
        return


    def check_params(self, side, e_target, e_stop_loss, time):
        close = self.sample[time][0]
        target = close * (1 + side * e_target)
        stop_loss = close * (1 - side * min(max_total_loss, e_stop_loss))
        rr = (target - close)/(close - stop_loss)
        if stop_loss < 0:
            stop_loss = 0
        if target < 0:
            target = 0
        e = -(stop_loss / close - 1)
        #leverage = round(max_total_loss / (abs(e) + fee_rate * (abs(e) + 2)), 0)
        leverage = LEVERAGE
        if leverage > MAX_LEVERAGE:
            leverage = MAX_LEVERAGE
        elif leverage <= 1:
            leverage = 1
            #return False
        #if rr > 7:
        #  return False
        if abs(e) >= min_e and leverage >= 1 and abs(close - stop_loss) > 2 * tick_size:
            entry_price = close
            order = Order(e, 0, 'ADA', target, entry_price, stop_loss, side, None)
            order.leverage = leverage
            order.rr = rr
            order.targets, order.stop_losses = [target], [stop_loss]
            order.start_time = time
            self.order_list.append(order)
            self.profit_t1 = 0
            self.profit_t = 0
            #order.index = ohlc.index[time]
            #print('e:', order.e)
            #print('leverage:', order.leverage)
            return True
        else:
            return False

    def show_orders(self):
        print('   e    | side |exit rr |leverage|  fee   | profit |   duration  |  max rr   ')
        for order in self.order_list:
            a = [round(order.e, 4), round(order.side, 4), round(order.exit_rr, 4), order.leverage,
                 round(order.fee, 4),
                 round(order.profit, 4), round(order.duration, 4), round(order.max_rr, 4)]

            for x in a:
                print(str(x) + str_n(' ', 8 - len(str(x))) + '|', end="")
            print()

    def visualize(self):
        plot.show_chart(self.sample, self.symbol, self.order_list)


class Order:
    list = []

    def __init__(self, e, max_rr, symbol, target, entry_price, stop_loss, side, time):
        # self.entry = entry
        self.e = e
        self.leverage = 0
        self.max_rr = max_rr
        self.exit_rr = 0
        self.profit = 0
        self.entry_fee_rate = taker_fee
        self.exit_fee_rate = taker_fee
        self.fee = 0
        self.duration = 0

        self.symbol = symbol
        self.target = target
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.stop_losses = [stop_loss]
        self.side = side
        self.time = time
        self.exit_price = 0
        self.lost = False
        self.won = False

    def calculate_leverage(self):
        leverage = round(
           max_total_loss / (abs(self.e) + self.entry_fee_rate + self.exit_fee_rate * (1 + self.e)), 0)
        if leverage > MAX_LEVERAGE:
            leverage = MAX_LEVERAGE
        self.leverage = leverage

    def calculate_profit(self):

        if self.entry_price == 0:
            self.calculate_leverage()
            rr = 0
            if self.max_rr >= rr and self.max_rr >= 1:
                self.exit_rr = rr
                self.exit_fee_rate = maker_fee
            elif self.max_rr >= 1:
                self.exit_rr = 0.2
                self.exit_fee_rate = taker_fee

            # elif self.max_rr >= 0.5:
            #    self.exit_rr = 0.1
            #    self.exit_fee_rate = taker_fee
            else:
                self.exit_rr = -1
                self.exit_fee_rate = taker_fee
        else:
            self.exit_rr = (self.exit_price - self.entry_price) / (self.entry_price - self.stop_losses[0])

        #if self.exit_rr >= self.exit_rr * 0.95:
        #    self.exit_fee_rate = maker_fee
        self.exit_fee_rate = taker_fee

        self.profit = self.leverage * (self.exit_rr * abs(self.e)) - self.calculate_fee()
        return self.profit

    def calculate_fee(self):
        # F = lf (2 + e)  F = l ( f1 + f2(1+e) )
        self.fee = self.leverage * (self.entry_fee_rate + self.exit_fee_rate * (1 + self.e))
        return self.fee


def calculate_total_profit(order_list):
    profit = 1
    for order in order_list:
        profit *= (1 + order.profit)

    return profit - 1


def str_n(ch, n):
    st = ""
    for i in range(n):
        st += ch
    return st


def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t
def softmax(vector):
 e = exp(vector)
 return e / e.sum()



