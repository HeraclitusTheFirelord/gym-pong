import gym
import numpy as np
import keras as ks
import time
import datetime
import random
import math

trainNew = False # 是否抛弃已训练好的网络
trainEpisode = 0 # 训练周期数（设为0表示不进行训练，直接展示结果）（每个周期不一定会跑完整个游戏）
trainCheckPoint = 1 # 训练多少周期输出一次成果
trainSaveEpisode = 10 # 训练多少周期保存一次参数
presentationTime = 10 # 展示训练成果时跑的游戏次数
presentationRender = False # 展示成果时是否render

class pongModel:
    ACTION_MEANING = [2, 5] # UP, DOWN
    OB_SIZE = [80, 80]
    def __init__(self):
        self.env = gym.make('Pong-v0')

    def reset(self):
        self.env.reset()
        observation, reward, done, info = self.env.step(0)
        self.ob0 = self.ob1 = observation
        return self.getObservation()

    def render(self):
        self.env.render()

    def step(self, _action):
        action = self.ACTION_MEANING[_action]
        observation, reward, done, info = self.env.step(action)
        self.ob1 = self.ob0
        self.ob0 = observation
        return self.getObservation(), reward, done, info

    def getEnv(self):
        return env

    def getObservation(self):
        gray0 = 0.2989 * self.ob0[34:194,:,0] + 0.5870 * self.ob0[34:194,:,1] + 0.1140 * self.ob0[34:194,:,2]
        gray1 = 0.2989 * self.ob1[34:194,:,0] + 0.5870 * self.ob1[34:194,:,1] + 0.1140 * self.ob1[34:194,:,2]
        return np.array(gray0[::2,::2] - gray1[::2,::2]) # shape (80, 80)

class learningModel:
    learningRate = 0.0001
    gamma = 0.99
    filePath = "modelWeights.h5"
    limitMemorySize = 1000

    def __init__(self, trainNew = True):
        self.model = ks.models.Sequential()
        self.model.add(ks.layers.Reshape((80,80,1), input_shape=(pongModel.OB_SIZE[0], pongModel.OB_SIZE[1])))
        
        self.model.add(ks.layers.Conv2D(4, 5, activation='relu'))
        self.model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(ks.layers.Conv2D(4, 5, activation='relu'))
        self.model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(128, activation = 'relu',))
        self.model.add(ks.layers.Dense(2, activation='softmax'))

        opt = ks.optimizers.RMSprop(lr=self.learningRate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt)
        
        if trainNew == False:
            self.load()
        self.resetMemory()

    def save(self):
        print("Saving weights...")
        self.model.save_weights(self.filePath)
        print("Weights saved.")

    def load(self):
        print("Loading weights...")
        self.model.load_weights(self.filePath)
        print("Weights loaded.")

    def getActionProb(self, ob):
        return self.model.predict(ob.reshape([1, pongModel.OB_SIZE[0], pongModel.OB_SIZE[1]]), batch_size=1).flatten()

    def resetMemory(self):
        self.trainState = []
        self.trainGradient = []
        self.trainReward = []
        self.trainProb = []
        self.trainMemorySize = 0

    def remember(self, state, gradient,  reward, prob):
        self.trainState.append(state)
        self.trainGradient.append(gradient)
        self.trainReward.append(reward)
        self.trainProb.append(prob)
        self.trainMemorySize += 1

    def intialTrain(self):
        trainX = np.array(self.trainState)
        # rewards
        rewards = np.zeros_like(self.trainReward)
        for i in range(len(self.trainReward)-1, -1, -1):
            if self.trainReward[i] != 0:
                reward = self.trainReward[i]
            else:
                reward = reward * self.gamma
            rewards[i] = reward
        rewards = np.vstack(rewards)
        rewards = rewards - np.mean(rewards)
        rewards = rewards / np.std(rewards)
        #gradients
        gradients = np.vstack(self.trainGradient) * rewards
        trainY = self.trainProb + self.learningRate * gradients
        return trainX, trainY

    def train(self):
        trainX, trainY = self.intialTrain()
        self.model.train_on_batch(trainX, trainY)
        self.resetMemory()

    def presentation(self, game, preNum, preRender):
        if preNum == 0:
            return
        print("Begin presentation")
        bestReward = -21
        sumReward = 0
        for i_episode in range(preNum):
            observation = game.reset()
            totalReward = 0
            for step in range(10000):
                if preRender:
                    game.render()
                #time.sleep(0.01)
                prob = self.getActionProb(observation)
                action = 0 if random.uniform(0, 1) < prob[0] else 1
                observation, reward, done, info = game.step(action)
                totalReward += reward
                if done:
                    sumReward += totalReward
                    if totalReward > bestReward:
                        bestReward = totalReward
                    print("Episode {} reward: {}".format(i_episode+1, totalReward))
                    break
        print("best: {}, average: {}".format(bestReward, sumReward / preNum))

def main():
    model = learningModel(trainNew=trainNew)
    game = pongModel()
    totalEpisode = trainEpisode
    checkpointEpisode = trainCheckPoint
    saveEpisode = trainSaveEpisode
    checkTime = datetime.datetime.now()
    for i_episode in range(totalEpisode):
        observation = game.reset()
        win = 0
        lose = 0
        for step in range(50000):
            #game.render()
            #time.sleep(0.01)
            prob = model.getActionProb(observation)
            action = 0 if random.uniform(0, 1) < prob[0] else 1
            gradient = np.zeros(2, dtype=float)
            gradient[action] = 1
            gradient = gradient - prob
            tmpObservation = observation
            observation, reward, done, info = game.step(action)
            model.remember(tmpObservation, gradient, reward, prob)
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            if done or (reward != 0 and model.trainMemorySize > model.limitMemorySize):
                model.train()
                if (i_episode+1) % checkpointEpisode == 0:
                    nowTime = datetime.datetime.now()
                    passTime = nowTime - checkTime
                    timePassed = passTime.seconds + passTime.microseconds / 1000000
                    print("Episode {} consequence: {}-{}, time: {}".format(i_episode+1, win, lose, timePassed))
                    checkTime = datetime.datetime.now()
                if (i_episode+1) % saveEpisode == 0:
                    model.save()
                break
    model.presentation(game, presentationTime, presentationRender)

if __name__ == '__main__':
    main()