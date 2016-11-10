
### **Motivation behind the paper** 
- DNNs (CNNs) achieved impressive results on various difficult learning tasks but they don't perform well on mapping sequences to sequences.
- They need a huge amount of labeled data
- They can only be applied to problems where both the inputs and outputs are representable by a fixed dimension vector. This is the most
important drawback of DNNs and motivation behind RNNs, simply because in some problems a fixed input/output size cannot be determined.

### **Sequence to Sequence idea**
- Train an LSTM to map the input sequence to a fixed size vector (encode)
- Train another LSTM to spit the target output from that vector (decode)

##### Warm-up 1: What is a sequence?
By definition a sequence is a set of related items, that follow each other in a specific order or according to a certain formula.
From highschool math remember, geometric sequences and arithmetic ones. The types of sequences we are talking about here is not very
different. We might not know the underlying formula or how the order came to exist but we do know the putting an h, e, l, l,o after one
another in this specific order, results in a meaningful word. Letters, Words, Numbers, Music notes Anything with such characterstic
should be a good candidate for sequence learning.

##### Warm-up 2: one step less than a seq2seq model?
You can think of a language model as half a seq2seq model. It uses only one RNN to learn a distribution of sequences of words in a
natural language. This allows the model to make next word prediction given previous words (context)(1)
http://www.scholarpedia.org/article/Neural_net_language_models#The_mathematics_of_neural_net_language_models


##### Warm-up 3: What is an LSTM? (simple language model case)
This paper uses an LSTM RNN unit in order to build the suggested model. LSTMs re introduced in (2) http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
A good resource to learn LSTMS is Colah's famous blog post http://colah.github.io/posts/2015-08-Understanding-LSTMs/
As a summary, you can think of an LSTM as a single pipe. This pipe has 4 gates. Each time, the pipe take an input (ex: word/letter) and
outputs a hypothesis. They hyopthesis is used to train the LSTM (by comparing it to the gold output during training)
or to make a prediction after the LSTM has been trained. An LSTM is therefore, capable of handle long term dependencies between input and output.
Without the gates, information flows through the pipe unchanged. This is called the cell state. Cell state usually holds context
information, such as current subject's gender etc.
The first gate is the forget gate, which looks at the previous hypothesis and the current input and outputs a number between
0 (get rid of it)and 1 (keep it).
Secondly, we need to decide what new info to add to our current flow. This is done using two gates. One gate, a sigmoid layer, decides which values we will update.
the second gate, a tanh layer, creates a vector of new possible values, that we could combine to our new state.
Now we are ready to update our cell state. we multiply the old state by the forget gate values. Then we add the new values from the 3rd gate, multiplied with
how much we decided to update each value (2nd gate).
The final gate decides the ouput. This is divided into two layers. A sigmoid layer that decides wich part of the cell state we will output and a tanh layer which
pushes the values to [-1, +1] range. 
Therefore, in each time step (with each input) the LSTM cell, changes its current state to keep up with the context and outputs a
hypothesis that helps in predicting the next input. 


### **Introduction** 
The paper answeres the previously mentioned motivations by introducing a sequence to sequence mapping between input and output. for
example, question answering, machine translation and speech recognition.
The idea of the model: Use one LSTM to read the input sequence, one timestep at a time --> represent it with a fixed dimensional vector.
then use another LSTM to extract the output sequence given that vector(langauge model)
input --LSTM1--> input_vec --LSTM2--> output
An important contribution of this work is the idea of reversing the order of words in the input sequence. This introduces short term 
dependencies between the input sentence and target one.


### **The model** 
Given a sequence of inputs x1,....xt a standard RNN computes y1,...yT by iterating:

 insert eq1
the goal of the LSTM is to estimate the conditional probability p(y1,...yT|x1,...xt) note that T might differ from t (different
input/output sequence length -recall motivation-). This is done by multiplying the probabilities from k=1:T of getting an output yk 
given the input's vector representation v and the previous output words.
 insert eq2


**model details:**

1. Use 2 LSTMs, one for input sequence other for output.
2. Use 4 layer LSTM.
3. Reverse the order of the words of the input sequence for example a b c is reversed to c b a and fed to the LSTM in this order. This 
makes the word a closer to its relevent ouput which improves the overall performance.



### **Experiences** 
WMT’14 English to French MT task and rescoring a n-best list of a SMT baseline system.
- **dataset**

 WMT’14 English to French dataset. 12M sentences, 348M French words and 304M English words
-**Vocab size**

 160K English, 80K French, <UNK> for every out-of-vocab word.
- **Training objective**

 max the log probability of a correct translation T given a source sentence S
insert eq3
- **decoding**

 They find the most likely translation by extracting the max arg according to the probabilities  assigned by the LSTM

 They use a simple beam search decoder which hold a B number of partial hypotheses, these are prefixes to some translations. at each timestep they extend a hypothesis with every possible word in vocab. This can get very big so they only keep the B most likely ones. As soon as they hit an <eos> symbol the corresponding hypothesis is removed from the beam and added to the complete hypotheses
For rescoring the baseline system,they compute the log probability of each hypothesis with the LSTM and avg their score with the LSTM's score 

- **reversing the source sentences**

 reversing source sentences (not target ones) the LSTM's test perplexity increased from 5.8 to 4.7 and the test BLEU
score of its decoded translations increased from 25.9 to 30.6
- **Training details**

 * deep LSTMs with 4 layers, with 1000 cells at each layer and 1000 dimensional word embeddings.
 * a naive softmax over 80,000 words at each output.
 * The resulting LSTM has 380M parameters of which 64M are pure recurrent connections (32M for the “encoder” LSTM and 32M for the “decoder” LSTM).
 * LSTM’s parameters initialized with the uniform distribution between -0.08 and 0.08 
 * lr = 0.7 (after 5 epochs, halved every half epoch)
 * 7.5 epochs
 * batch size = 128
 * gradient clipping (to manage exploding gradients). when norm of the gradient exceeds a threshold. For each training batch, we compute
s = kgk2, where g is the gradient divided by 128. If s > 5, we set g = 5g
s.
 * sentences withing the same mini batch are roughly the same length (buckets method)
 * Parallized over 8 GPUs

### **Results** 











