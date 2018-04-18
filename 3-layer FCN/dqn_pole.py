import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import DQN_model4 as DQN
import gym
import collections
import random
max_ep = 10000
env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]

output_size = env.action_space.n
print(input_size,"  ",output_size)
def simple_replay_Train(DQN,train_batch): # 플레이한 데이터를 바탕으로 dqn을 학습
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state,action,reward,next_state,done in train_batch:
        Q = DQN.predict(state)

        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + 0.9*np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])

    return DQN.update(x_stack,y_stack)

def bot_play(mainDQN): # dqn을 학습시키기위한 train_batch 를 만들기위해 플레이
    # 게임 1판 진행
    # 게임오버 또는 스텝이 100번 될때까지 게임플레이
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s,reward,done,_ = env.step(a)
        reward_sum += reward
        if done:
            print("[ PLAY  ] Total Score: {}".format(reward_sum),end = "  ")
            break

def main(): # 메인함수
    replay_buffer = collections.deque()
    autosave =100 # 모델 자동세이브 주기 (episode)

    with tf.Session() as sess:
        mainDQN = DQN.DQN(sess,input_size,output_size,"main")
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        try: # 모델 로드 시도
            pre_episoce = 2201
            saver.restore(sess, "./model4/"+str(pre_episoce)+"/"+str(pre_episoce)+".ckpt")
            print("Restore model")
            time.sleep(1)
        except: # 실패시 종료
            print("newmodel")
            pre_episoce = 0

        for episode in range(pre_episoce+1 ,max_ep): # 에피소드 루프 시작
            #e =1./((episode/10)+1) # 에피소드가 커질수록 dqn을 많이참조
            done = False
            step_count = 0
            state = env.reset()
            while not done:

                if np.random.rand(1)*50  < episode:
                    action = env.action_space.sample()
                else :
                    action = np.argmax(mainDQN.predict(state))

                next_state,reward,done,_ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state,action,reward,next_state,done))
                if (len(replay_buffer)>50000):
                    replay_buffer.popleft()

                state = next_state
                step_count +=1
                if step_count >10000:
                    break

                if episode % 10 == 1 and len(replay_buffer) > 100:   # 10번중에 1번의 경우 데이터를 바탕으로 학습하면서 게임진행
                    loss = 0
                    #for _ in range(10):
                    minbatch = random.sample(replay_buffer,100)
                    loss,_ = simple_replay_Train(mainDQN,minbatch)
                    print("[ LEARN ] step: %d loss: %f"%(step_count,loss))


            bot_play(mainDQN) # 1판 플레이
            print("episodes: {} max_steps : {}".format(episode, step_count))

            if episode % autosave == 1: # 자동세이브
                try:
                    save_path = saver.save(sess, "./model4/" + str(episode) + "/" + str(episode) + ".ckpt")
                    print("Model saved in file: %s" % save_path)
                except:
                    print("fail_2")

if __name__ == '__main__':
    main()