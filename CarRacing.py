import numpy as np
import tensorflow as tf
import base64, io, time , gym, os
import IPython, functools
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
from tqdm import tqdm


class LossHistory:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss

class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()



env = gym.make("CarRacing-v0")
env.seed(1)

n_observations = env.observation_space
n_actions = env.action_space.shape[0]


def create_carracing_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32,5,strides= 2, input_shape =(84,84,1)))
  model.add(tf.keras.layers.LeakyReLU())

  model.add(tf.keras.layers.Conv2D(64,5,strides= 2))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Flatten())
  
  model.add(tf.keras.layers.Dense(3, activation = 'tanh'))
  return model

def preprocess_image(image):
    image = image[:84,:84,1].astype('float32')
    image = (image-127.5)/127.5
    return image.reshape((84,84,1) )


def choose_action(model, observation):
    observation = np.expand_dims(observation, axis =0)
    action = model.predict(observation)
    return action[0,:]
##############
##till here
##############
class Memory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

memory = Memory()

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma = 0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        
        R = R*gamma + rewards[t]
        discounted_rewards[t] = R
    
    return normalize(discounted_rewards)

msle = tf.keras.losses.MeanSquaredLogarithmicError()

def compute_loss(logits, actions, rewards):
    #neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = actions)
    neg_logprob =  msle(logits, actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

def train_step(model, optimizer, observations, actions , discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



#params 
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# instantiate car agent
cartpole_model = create_carracing_model()
smoothed_reward = LossHistory(smoothing_factor=0.9)
plotter = PeriodicPlotter(sec =5, xlabel='Iterations', ylabel='Rewards')


# to track our progres
checkpoint_dir = "./carracing_training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
checkpoint = tf.train.Checkpoint(model=cartpole_model  ,optimizer = optimizer)


if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

try:
    for i_episode in range(500):
    
    #plotter.plot(smoothed_reward.get())
    
        # Restart the environment
        observation = env.reset()
        previous_frame = preprocess_image(observation)
        memory.clear()

        while True:
            # using our observation, choose an action and take it in the environment
            env.render()

            current_frame = preprocess_image(observation)
            obs_change = current_frame - previous_frame

            action = choose_action(cartpole_model, obs_change)
            next_observation, reward, done, info = env.step(action)
            # add to memory
            #next_observation = preprocess_image(next_observation)


            memory.add_to_memory(obs_change, action, reward)
            
            # is the episode over? did you crash or do so well that you're done?
            if done:
                # determine total reward and keep a record of this
                total_reward = sum(memory.rewards)
                smoothed_reward.append(total_reward)
                
                # initiate training - remember we don't know anything about how the 
                #   agent is doing until it has crashed!
                train_step(cartpole_model, optimizer, 
                            observations=np.array(memory.observations),
                            actions=np.array(memory.actions),
                            discounted_rewards = discount_rewards(memory.rewards))
                
                # reset the memory
                memory.clear()
                break
            # update our observatons
            observation = next_observation
            previous_frame = current_frame
except:
    checkpoint.save(file_prefix = checkpoint_prefix)


checkpoint.save(file_prefix = checkpoint_prefix)