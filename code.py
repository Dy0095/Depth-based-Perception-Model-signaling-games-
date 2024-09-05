# Main part of the depth-based perception model
# to adjust the frequencies of states, see generate_state.py and change FrequencyA, MutantA and MutantB
# to adjust the payoff matrix, see function payoff(state, action)
# to adjust the cost function, see function cost_perception(m)

import random
import matplotlib.pyplot as plt
import os
import xlsxwriter


FrequencyA = 50    # decide the frequencies of X (including X1 and X2).
MutantA = 50    # decide the frequencies of X1
MutantB = 50    # decide the frequencies of Y1
State = []
StateNumber = 1000
i_state = 1


def generate_state():
    global i_state
    while i_state <= StateNumber:
        if random.randint(1, 100) <= FrequencyA:
            if random.randint(1, 100) <= MutantA:
                State.append([1,3,5])
                i_state += 1
            else:
                State.append([1,3,7])
                i_state += 1
        else:
            if random.randint(1, 100) <= MutantB:
                State.append([2,3,5])
                i_state += 1
            else:
                State.append([2,4,5])
                i_state += 1


perception_depth_weight_urn = [1, 1, 1]
# urn for choosing which depth, in our model we just let the initial weight of every depth be 1 for simplicity
# the mechanism of inventing new signals introduced in Skyrms(2010) can also be applied here
i = 1
PerceivedState = []     # current perceived state up to the chosen depth
perceived_state_memory = []     # perceived states that the organism has stored
perceived_state_urn = []
# combination of all perceived state urns, each contains the weights of four internal signals
signal_urn = [[1, 1], [1, 1], [1, 1], [1, 1]]
# combination of all four internal signal urns, each contains the weights of two actions
state_type = [[1, 3, 5], [1, 3, 7], [2, 3, 5], [2, 4, 5]]       # represent X1 X2 Y1 Y2 respectively
action_type = [1, 2]    # represent eat or ignore
total_payoff = 0    # accumulated net payoff
total_payoff_record = []    # used for plot
perception_depth_weight_record = [[],[],[]]     # used for plot
perception_depth_weight_percent_record = [[],[],[]]     # used for plot


# to make sure the ball in the urn will not be smaller than 1 when punishment is permitted
def one_plus(m, y):
    if m + y >= 1:
        return m + y
    else:
        return 1


def payoff(state, action):      # payoff matrix, X1,X2,Y1,Y2 corresponds to 0、1、2、3，ingest、avoid with 0,1
    if state == 0:
        if action == 0:
            return 3    # X1, eat
        else:
            return -2   # X1, ignore
    elif state == 1:
        if action == 0:
            return -1   # X2, eat
        else:
            return 2    # X2, ignore
    elif state == 2:
        if action == 0:
            return -2   # Y1, eat
        else:
            return 3    # Y1, ignore
    else:
        if action == 0:
            return 2    # Y2, eat
        else:
            return -1   # Y2, eat
# origin:       3 -2 -1 2 -2 3 2 -1
# 2-partition:  3 -2 -1 1 -2 3 0 -1
# 3-partition:  1.5 -1 -1 0 -3.5 3 3 -3.5
# 4-partition:  3 -3 -3 3 -3 3 3 -3


def cost_perception(m):         # cost function
    return m - 0.8
# origin cost: m - 0.8
# 2-partition: 0.5 * m - 0.5
# 3-partition: 0.1 * pow(m - 1, 4) + 0.65
# 4-partition: 0.2 * m + 0.8


def execute_signaling():    # represent 1 simulation
    global i
    while i <= StateNumber:
        depth_based_signaling()


def depth_based_signaling():    # choose depth
    global i, PerceivedState
    if random.random() < perception_depth_weight_urn[0] / (sum(perception_depth_weight_urn[:])):  # depth 1
        PerceivedState = [State[i - 1][0]]  # perceive the external state up to the chosen depth
        signal_in_depth(1)  # execute the previous part with depth 1
    elif random.random() < perception_depth_weight_urn[1] / (sum(perception_depth_weight_urn[:])):  # depth 2
        PerceivedState = [State[i - 1][0], State[i - 1][1]]
        signal_in_depth(2)
    else:  # depth 3
        PerceivedState = [State[i - 1][0], State[i - 1][1], State[i - 1][2]]
        signal_in_depth(3)


def signal_in_depth(x):         # after decide a perceptual depth
    global perceived_state_memory, perceived_state_urn, i, j, total_payoff, total_payoff_record
    if perceived_state_memory.count(PerceivedState) == 0:  # if that perceived state has not been stored in memory
        perceived_state_memory += [PerceivedState]  # record that perceived state in memory
        j = perceived_state_memory.index(PerceivedState)    # record its index number
        perceived_state_urn += [[1, 1, 1, 1]]  # add a new urn for that state with all initial weight set as 1
        # their index numbers are same
        p = random. random()
        perceived_state_urn_probability = []
        c = 0
        while c < len(perceived_state_urn[j]):
            perceived_state_urn_probability += [sum(perceived_state_urn[j][0:c])/sum(perceived_state_urn[j])]
            # consists of a probabilistic ordering of each weight and the sum of the previous weights
            # i.e., 33.3%, 66.7%, and 1 in the case of 1,1,1, respectively
            # which ensures that the order is still the same after reordering
            c += 1
        perceived_state_urn_probability.append(p)
        perceived_state_urn_probability = sorted(perceived_state_urn_probability)
        sg = perceived_state_urn_probability.index(p)-1    # position of p represents which signal is selected
        q = random. random()
        urn_action_probability = []
        d = 0
        while d < len(signal_urn[sg]):
            urn_action_probability += [sum(signal_urn[sg][0:d])/sum(signal_urn[sg])]
            d += 1
        urn_action_probability.append(q)
        urn_action_probability = sorted(urn_action_probability)
        ac = urn_action_probability.index(q) -1      # represent which action the organism choose，0 for eat，1 for ignore
        st = State[i - 1]
        # the following part is to adjust the weights in the corresponding urns and record other value
        signal_urn[sg][ac] = one_plus(signal_urn[sg][ac], payoff(state_type.index(st), ac) - cost_perception(x))
        perceived_state_urn[j][sg] = one_plus(perceived_state_urn[j][sg], payoff(state_type.index(st), ac) - cost_perception(x))
        perception_depth_weight_urn[x - 1] = one_plus(perception_depth_weight_urn[x - 1], payoff(state_type.index(st), ac) - cost_perception(x))
        total_payoff += payoff(state_type.index(st), ac) - cost_perception(x)
        total_payoff_record += [total_payoff]
        z0 = 0
        for z in perception_depth_weight_urn:
            percent0 = 100 * z/(sum(perception_depth_weight_urn[:]))
            perception_depth_weight_record[z0] += [z]
            perception_depth_weight_percent_record[z0] += [percent0]
            z0 += 1
        i += 1
    else:  # if that perceived state has been stored in memory
        j = perceived_state_memory.index(PerceivedState)    # search the position of the stored perceived state
        # the following part is the same as the previous part
        p = random. random()
        perceived_state_urn_probability = []
        c = 0
        while c < len(perceived_state_urn[j]):
            perceived_state_urn_probability += [sum(perceived_state_urn[j][0:c])/sum(perceived_state_urn[j])]
            c += 1
        perceived_state_urn_probability.append(p)
        perceived_state_urn_probability = sorted(perceived_state_urn_probability)
        sg = perceived_state_urn_probability.index(p)-1
        q = random. random()
        urn_action_probability = []
        d = 0
        while d < len(signal_urn[sg]):
            urn_action_probability += [sum(signal_urn[sg][0:d])/sum(signal_urn[sg])]
            d += 1
        urn_action_probability.append(q)
        urn_action_probability = sorted(urn_action_probability)
        ac = urn_action_probability.index(q) -1
        st = State[i - 1]
        signal_urn[sg][ac] = one_plus(signal_urn[sg][ac], payoff(state_type.index(st), ac) - cost_perception(x))
        perceived_state_urn[j][sg] = one_plus(perceived_state_urn[j][sg], payoff(state_type.index(st), ac) - cost_perception(x))
        perception_depth_weight_urn[x - 1] = one_plus(perception_depth_weight_urn[x - 1], payoff(state_type.index(st), ac) - cost_perception(x))
        total_payoff += payoff(state_type.index(st), ac) - cost_perception(x)
        total_payoff_record += [total_payoff]
        z0 = 0
        for z in perception_depth_weight_urn:
            percent0 = 100 * z/(sum(perception_depth_weight_urn[:]))
            perception_depth_weight_record[z0] += [z]
            perception_depth_weight_percent_record[z0] += [percent0]
            z0 += 1
        i += 1


r = 999     # the number of simulations
q = 1
iter_payoff_record = []     # for plot
name_list = []
figure_save_path = 'figure'    # path of the outputs
# don't forget to create a folder named as figure_save_path if you want to rename the path
# same for the name of excel
x_loc = 1
wb = xlsxwriter.Workbook('data.xlsx')     # excel for recording the data in the simulation
ws = wb.add_worksheet('depth-based perception model')
for s in range(r):
    name_list += ['depth-based perception model' + str(s+1)]


def iter_depth_based_signaling():       # run simulation for r times
    global i_state, q, r, x_loc, iter_payoff_record, perception_depth_weight_urn, PerceivedState, perceived_state_memory, perceived_state_urn, signal_urn, total_payoff, total_payoff_record, perception_depth_weight_record, i, perception_depth_weight_percent_record
    while q <= r:
        i_state = 1
        generate_state()
        # to reset all data
        perception_depth_weight_urn = [1, 1, 1]
        PerceivedState = []
        perceived_state_memory = []
        perceived_state_urn = []
        signal_urn = [[1, 1], [1, 1], [1, 1], [1, 1]]
        total_payoff = 0
        total_payoff_record = []
        perception_depth_weight_record = [[], [], []]
        perception_depth_weight_percent_record = [[], [], []]
        i = 1
        execute_signaling()
        for wt in range(len(perception_depth_weight_urn)):
            ws.write(x_loc,wt,perception_depth_weight_urn[wt])
        x_loc += 1

        '''
        # this part is used to record the data of every simulation for further analysis
        for wt in range(len(perceived_state_memory)):
            ws.write(x_loc, wt, str(perceived_state_memory[wt]))
            ws.write(x_loc+1,wt,perceived_state_urn[wt][0])
            ws.write(x_loc+2,wt,perceived_state_urn[wt][1])
            ws.write(x_loc+3,wt,perceived_state_urn[wt][2])
            ws.write(x_loc+4,wt,perceived_state_urn[wt][3])
        for wt in range(len(perception_depth_weight_urn)):
            ws.write(x_loc+5,wt,perception_depth_weight_urn[wt])
        x_loc += 6
        for wt in range(len(signal_urn)):
            ws.write(x_loc, wt, signal_urn[wt][0])
            ws.write(x_loc+1, wt, signal_urn[wt][1])
        x_loc += 4
        '''

        # the part of plotting every single simulation
        plt.figure(figsize=(18, 9))
        plt.subplot(121)
        plt.plot(total_payoff_record)
        plt.ylabel('total payoff')
        plt.subplot(122)
        plt.plot(range(1000), perception_depth_weight_percent_record[0], 'r-', range(1000), perception_depth_weight_percent_record[1], 'b-', range(1000), perception_depth_weight_percent_record[2], 'y-')
        plt.ylabel('Relative Weight')
        plt.xlabel('Step')
        plt.savefig(os.path.join(figure_save_path,name_list[q-1]))
        plt.close()
        iter_payoff_record += [total_payoff_record]
        q += 1


iter_depth_based_signaling()
wb.close()
# to plot the median accumulated net payoff
median_payoff_record = []
unsorted_payoff_record = []
sorted_payoff_record = []
for x in range(StateNumber):
    unsorted_payoff_record += [[]]
    sorted_payoff_record += [[]]
for x in range(StateNumber):
    for y in range(r):
        unsorted_payoff_record[x] += [iter_payoff_record[y][x]]
for x in range(StateNumber):
    sorted_payoff_record[x] = sorted(unsorted_payoff_record[x])
    median_payoff_record += [sorted_payoff_record[x][499]]
plt.plot(median_payoff_record)
plt.ylabel('Accumulated Net Payoff')
plt.xlabel('Step')
plt.savefig(os.path.join(figure_save_path, 'median_total_payoff'))