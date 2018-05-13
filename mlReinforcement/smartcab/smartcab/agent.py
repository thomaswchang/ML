import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    # Epsilon: aka exploration factor:
    #       Decay function that controls how many ITERATIONS is used to form the Q table.
    #       Once the value becomes lower than a THRESHOLD value, we switch from exploration phase into learning phase
    # Alpha: aka learning factor:
    # Discount factor : aka gamma, controls how much the previous Q table affects our current iteration
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.trialCount = 0      # Used to denote which trial we are at

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.training_cnt = -1


    def reset(self, destination=None, testing=False, trial=1):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed.
            Env's reset method calls this method, and sets the testing parameters"""

        if not testing:
            self.trialCount = self.trialCount + 1
            print ("Increment trial count")

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            #self.training_cnt += 1

            # Learning rate; determines how much weight needs to be given to current rewards over the previously learned values.
            # This determines how fast the model can converge (a smaller value would mean that the agent would take a higher number of trials to reach the optimal policy,
            # but it does learn a better policy). Hence, it should be tuned separately.

            # Exploration rate
            # Determines how much we explore.

            #self.epsilon = 1 / (1 + math.exp(0.006 * (trial - 30))) + 0.04


            self.epsilon = 1 / (1 + math.exp(0.006 * (trial - 30)))

            print "TWC: agent.py reset(): cnt={}".format(trial)

            # Results using self.epsilon = 1 / (1 + math.exp(0.005 * (trial)))
            # Safety=F  Reliability=A+  trails=600

            # Results using self.epsilon = math.pow(0.9, self.trialCount)
            # alpha=1       epsilon=0.3     ==> Safety=F, Reliability=C
            # alpha=0.3     epsilon=0.9     ==> Safety=F, Reliability=B   29 trials


        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint = direction agent should move: forward, left, right
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
                                                # {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
                                                # light : green, red
                                                # oncoming : forward, left, right, None
                                                # left : forward, left, right, None
                                                # right : forward, left, right, None
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO
        ###########
        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engine,ered features, this learning process gets entirely negated.
        print "TWC: waypoint={} inputs={} deadline={}".format(waypoint, inputs, deadline)

        # Set 'state' as a tuple of relevant data for the agent
        # 4 * 4 * 4 * 2 * 3 = 384 states
        #state = (inputs['left'], inputs['right'],  inputs['oncoming'], inputs['light'], waypoint)
        state = (waypoint, tuple(sorted(inputs.items())))
        print "TWC: states={}".format(state)
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        """
        # Calculate the maximum Q-value of all actions for a given state
        allPossibleStatesAndItsQDict = self.Q[state]
        import operator
        maxQ = 0
        maxElements = max(allPossibleStatesAndItsQDict.iteritems(), key=operator.itemgetter(1))
        if len(maxElements) >0:
            maxElements[0]
        """
        maxQ = max(self.Q[state].values())
        return maxQ

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if (self.learning):
            if state not in self.Q:
                # Create a new dictionary where key are the actions, value are 0
                self.Q[state] = dict(zip(self.valid_actions, [0] * len(self.valid_actions)))
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        if self.learning:
            if self.epsilon > random.random(): # We want to explore ..
                action = random.choice(self.valid_actions) # WOW prett
            else:
                maxQ = self.get_maxQ(state)
                actionCandidates = []
                for keyAction in self.Q[state]:
                    if self.Q[state][keyAction] == maxQ:
                        actionCandidates.append(keyAction)
                action = random.choice(actionCandidates)
                print "TWC: action={}".format(action)
        else:
            action = random.choice(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            newQ = reward * self.alpha + (1-self.alpha) * self.Q[state][action]
            self.Q[state][action] = newQ
            print "TWC: learn(): update Q[state={}][action={}= to {}".format(state, action, newQ)
            if self.Q[state][action] > 0:
                print "Stop"
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    #agent = env.create_agent(LearningAgent)  # no learning
    agent = env.create_agent(LearningAgent, learning=True, epsilon=1.0, alpha=0.5) #TWC modified
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, display=True, log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10) #TWC: Once learning decaying function == threshold 0.05, we kick off testing, which is defined for 10.


if __name__ == '__main__':
    run()
