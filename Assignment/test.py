import sys
import time
import traceback

from environment import CliffBoxPushingBase


if __name__ == '__main__':
    env = CliffBoxPushingBase()
    env.reset()
    teminated = False
    time_step = 0
    env.print_world()

    rewards = []
    try:
        while not teminated:
            action = [int(v) for v in \
                input("Please input the actions (up: 1, down: 2, left: 3, right: 4): ").split()]
            
            reward, teminated, _ = env.step(action)
            rewards.append(reward)
            print(f'step: {time_step}, actions: {action}, reward: {reward}')
            env.print_world()
            
            time.sleep(0.5)
            time_step += 1

        print(f'rewards: {sum(rewards)}')
        print(f'print the historical actions: {env.episode_actions}')
    except:
        print('Something wrong happened..... print the historical actions')
        print(env.episode_actions)
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

        print("Exception type : %s " % ex_type.__name__)
        print("Exception message : %s" %ex_value)
        print("Stack trace : %s" %'\n'.join(stack_trace))