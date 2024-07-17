import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time

class GameOfLife:
    def __init__(self, height, width, rules=None):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.rules = rules if rules else {"B": [3], "S": [2, 3]}

    def step(self):
        new_board = np.copy(self.board)
        for r in range(self.height):
            for c in range(self.width):
                num_alive_neighbors = np.sum(self.board[max(0, r-1):min(r+2, self.height), max(0, c-1):min(c+2, self.width)]) - self.board[r, c]
                if self.board[r, c] == 1 and num_alive_neighbors not in self.rules["S"]:
                    new_board[r, c] = 0
                elif self.board[r, c] == 0 and num_alive_neighbors in self.rules["B"]:
                    new_board[r, c] = 1
        self.board = new_board

class GolLM:
    def __init__(self):
        self.spacing=5
        self.input_radius=3
        self.output_convolution_kernel=[0.2,0.5,1,0.5,0.2] #for this, spacing should be 5
        self.output_reset_radius=7
        self.output_collection_radius=3
        self.halting_threshold=30
        self.width=50
        self.vocabulary="0123456789|" #| is the halting symbol

        self.n_symbols = len(self.vocabulary)
        self.vocab_list=list(self.vocabulary)
        self.vocab_to_list_idx=dict(zip(self.vocabulary,range(self.n_symbols)))
        self.io_cells_idxs =[self.spacing*i for i in range(1,self.n_symbols+1)]
        self.height = self.spacing*self.n_symbols + self.spacing + 1
        self.v2c=dict(zip(self.vocabulary, self.io_cells_idxs))
        self.c2v=dict(zip(self.io_cells_idxs, self.vocabulary))
        self.j_input=0
        self.j_output=self.width-1

        self.half_width=int(self.width/2)
        self.reward_dict={-3:2, -2:4 , -1:6, 0:8, 1:10, 2:12, 3:14} #reward goes from -3 to 3, 14 values
        self.reward_zone=np.zeros((14))


        self.gol=GameOfLife(self.height,self.width)

        self.output_accumulation=np.zeros((self.height))
        self.single_inference_step=0

        # Create a figure and axis object
        self.fig, self.ax = None, None

        #make assertions, throw error if not satisfied
        self.init_assertions()


    def init_assertions(self): # To avoid initialization with wrong parameter values
        assert self.width > 2*self.output_reset_radius, "Width must be greater than 2*output_reset_radius"
        assert self.width >= 50 , "Width must be at least 50"
        assert len(self.output_convolution_kernel)==self.spacing, "Output convolution kernel must have length equal to spacing"
        assert self.spacing%2==1, "Spacing must be odd"
        assert self.width%2==0, "Width must be even"

    def randomize_board(self,p):
        self.gol.board=np.random.choice([0,1],size=(self.height,self.width),p=[1-p,p])

    def random_mutation(self,p):
        self.mutation_mask=np.random.choice([0,1],size=(self.height,self.width),p=[1-p,p])
        self.gol.board=self.gol.board+self.mutation_mask
        self.gol.board[self.gol.board>1]=0

    def reset_output(self):
        self.gol.board[:,-self.output_reset_radius:]=0

    def reset_output_accumulation(self):
        self.output_accumulation=np.zeros((self.height))

    def tokenize_letter(self,x):
        return self.v2c[x]
    
    def tokenize_sentence(self,sentence):
        return [self.tokenize_letter(x) for x in sentence]

    def set_input(self,x):
        self.x=x
    
    def set_input_bc(self):
        self.gol.board[:,self.j_input]=0
        upper_dist=int((self.input_radius-1)/2)
        lower_dist=int((self.input_radius-1)/2+1)
        self.gol.board[self.x-upper_dist:self.x+lower_dist,self.j_input]=1

    def apply_reward(self,reward):
        if reward==None:
            self.reward_zone[:]=0
        else:
            n_rew_cells=self.reward_dict[reward]
            self.reward_zone[0:n_rew_cells]=1
            for i in range(0,n_rew_cells,2):
                self.reward_zone[i]=0
        self.gol.board[-1,:]=0
        self.gol.board[0,:]=0
        if reward!=None:
            self.gol.board[-1,self.half_width-7:self.half_width+7]=self.reward_zone
            self.gol.board[0,self.half_width-7:self.half_width+7]=self.reward_zone
            self.gol.board[-1,8:13]=1
            self.gol.board[0,8:13]=1

    def compute_output(self):
        convolved_accumulation=np.convolve(self.output_accumulation,self.output_convolution_kernel,mode='same')
        output=convolved_accumulation[self.io_cells_idxs]
        output=output/np.max(output)
        return output

    def step(self,reward=None,vismode=False):
        if self.single_inference_step==0:
            self.init_assertions()
            self.set_input_bc()
            self.apply_reward(reward)
        self.random_mutation(0.01)
        self.gol.step()
        self.apply_reward(reward)
        self.set_input_bc()
        self.single_inference_step+=1
        if vismode:
            self.view()
    
    def single_token_inference(self,x,vismode=False):
        self.single_inference_step=0
        self.reset_output()
        self.reset_output_accumulation()
        self.set_input(x)
        while True:
            self.step(vismode=vismode)
            s=np.sum(self.gol.board[:,self.j_output])
            if s>0:
                self.output_accumulation+=np.sum(self.gol.board[:,-self.output_collection_radius:],axis=1)
                self.reset_output()
                if np.sum(self.output_accumulation)>self.halting_threshold:
                    output=self.compute_output()
                    break
        return output
    
    def meta_learn(self,x,reward,vismode=False):
        self.single_inference_step=0
        self.reset_output()
        self.reset_output_accumulation()
        self.set_input(x)
        while True:
            self.step(reward,vismode)
            s=np.sum(self.gol.board[:,self.j_output])
            if s>0:
                self.output_accumulation+=np.sum(self.gol.board[:,-self.output_collection_radius:],axis=1)
                self.reset_output()
                if np.sum(self.output_accumulation)>self.halting_threshold:
                    output=self.compute_output()
                    break
        return output

    def temperature_sampling(self,logits,temperature):
        logits=logits/np.max(logits)
        probs=np.exp(logits/temperature)/np.sum(np.exp(logits/temperature))
        return np.random.choice(range(len(probs)),p=probs)

    def cross_entropy_loss(self,output,target): #not used as of now
        #output is logits
        #target is a token (letter)
        target_idx=self.vocab_to_list_idx[target]
        target_onehot=np.zeros((self.n_symbols))
        target_onehot[target_idx]=1
        loss=-np.sum(target_onehot*np.log(output))
        return loss

    def compute_reward(self,output,target): #this is used
        #output is logits
        #target is a token (letter)
        output_sf=np.exp(output)/np.sum(np.exp(output))
        target_idx=self.vocab_to_list_idx[target]
        reward=output_sf[target_idx]
        return reward

    def scale_reward(self,reward):
        reward=(reward-0.5)*6 #scale reward to [-3,3]
        reward=round(reward)
        return reward

    def infer(self,s,vismode=False,temperature=1): #inference for next token
        #s is not a sentence but a single token
        #inference
        token=self.tokenize_letter(s)
        output=self.single_token_inference(token,vismode)
        next_token=self.temperature_sampling(output,temperature)
        next_token_letter=self.vocab_list[next_token]
        return output,next_token_letter
    
    def learn(self,s,reward,vismode=False): #feedback meata learning
        #s is not a sentence but a single token
        #learning
        token=self.tokenize_letter(s)
        output=self.meta_learn(token,reward,vismode)
        return output
    
    
    def learn_over_sentence(self,sentence,vismode=False): #for self-supervised autoregressive learning
        generated_output=[]

        for token in sentence: #loop over input tokens
            output1,next_token_letter=self.infer(token,vismode)
            print(f"\nRan through token '{token}' and sampled next token '{next_token_letter}'")
            print(f"Output distribution: {output1}")
            #compute reward
            reward=self.compute_reward(output1,token)
            print(f"Reward = {reward}")
            scaled_reward=self.scale_reward(reward)
            print(f"Scaled reward = {scaled_reward}")
            output2=self.learn(token,scaled_reward,vismode)
            print(f"New output distribution: {output2}")
            generated_output.append(next_token_letter)

        return generated_output


    def multiple_token_inference(self,sentence,temperature,vismode=False): #for inference without learning
        generated_output=[]

        for token in sentence: #loop over input tokens
            output1,next_token_letter=self.infer(token,vismode,temperature)
            output2=self.learn(token,0,vismode)
            print(f"Ran through token {token}")

        print(f"Effectively sampled token: {next_token_letter}")
        generated_output.append(next_token_letter)

        while True: #autoregressive loop until halting
            output1,next_token_letter=self.infer(next_token_letter,vismode,temperature)
            output2=self.learn(next_token_letter,0,vismode)
            print(f"Effectively sampled token: {next_token_letter}")
            generated_output.append(next_token_letter)
            
            if next_token_letter=="|":
                break

        return generated_output


    def view(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
        self.ax.clear()  # Clear the previous plot
        self.ax.imshow(self.gol.board, cmap='gray', interpolation='nearest')
        self.ax.set_yticks(self.io_cells_idxs)
        self.ax.set_yticklabels(self.vocab_list)
        plt.draw()
        plt.pause(0.01)  # Pause to update the figure

    def preview_lm(self):
        self.preview_state=np.zeros((self.height,self.width))
        #make grey the cells that are input or output only
        self.preview_state[self.io_cells_idxs,self.j_input]=0.5
        self.preview_state[self.io_cells_idxs,self.j_output]=0.5
        plt.imshow(self.preview_state, cmap='gray', interpolation='nearest')
        plt.yticks(self.io_cells_idxs, self.vocab_list)
        plt.show()
        print(f"State space dimension: {self.height}x{self.width} = {self.height*self.width} cells")
        print(f"Vocabulary dimension: {self.n_symbols} ; Symbols : {self.vocab_list}")
        print(f"Spacing between input cells: {self.spacing}")