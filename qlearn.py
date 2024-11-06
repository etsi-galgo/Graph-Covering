import random 

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.visit_count = {}  # Track state-action visit counts

    def getQ(self, state, action):
        """Get the Q-value for a state-action pair, defaulting to 0 if unseen."""
        return self.q.get((state, action), 0)

    def learnQ(self, state, action, reward, value):
        """Update the Q-value for a state-action pair using Q-learning formula."""
        oldv = self.q.get((state, action), 0)  # Initialize unseen Q-values as 0
        self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        """Choose an action based on ε-greedy policy."""
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        # Exploration: choose random action with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            # Exploitation: choose the best action
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]

        if return_q:  # Optional: return the action and Q-values for debugging
            return action, q
        return action

    def learn(self, state1, action1, reward, state2): 
        """Perform the Q-learning update for a given transition."""
        maxqnew = max([self.getQ(state2, a) for a in self.actions])  # max Q(s', a')
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)
        
        # Update visitation count
        self.visit_count[(state1, action1)] = self.visit_count.get((state1, action1), 0) + 1
        
    def get_visitation_count(self, state, action):
        return self.visit_count.get((state, action), 0)       
