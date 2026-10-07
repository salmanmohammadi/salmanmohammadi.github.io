---
title: The theory of Proximal Policy Optimization implementations
layout: post
---
## Prelude

The aim of this post is to share my understanding of some of the conceptual and theoretical background behind implementations of the [Proximal Policy Optimisation](https://arxiv.org/abs/1707.06347) (PPO) reinforcement learning (RL) algorithm. PPO is widely used due to its stability and simplicity - popular applications include [beating the Dota 2 world champions](https://openai.com/research/openai-five-defeats-dota-2-world-champions) and [aligning language models](https://arxiv.org/pdf/2203.02155.pdf). While the PPO paper provides quite a general and straightforward overview of the algorithm, modern implementations of PPO use several additional techniques to achieve state-of-the-art performance in complex environments{% sidenote "0" "[Procgen, Karle Cobbe et al.](https://openai.com/research/procgen-benchmark)<br>[Atari, OpenAI Gymnasium](https://gymnasium.farama.org/environments/atari/)" %}. You might discover this if you try to implement the algorithm solely based on the paper. I try and present a coherent narrative here around these additional techniques.

I'd recommend reading parts [one](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), [two](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html), and [three](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) of [SpinningUp](https://spinningup.openai.com/en/latest/index.html) if you're new to reinforcement learning. There are a few longer-form educational resources that I'd recommend if you'd like a broader understanding of the field{% sidenote "1" "- [A (Long) Peek into Reinforcement Learning, Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)<br> - [Artificial Intelligence: A Modern Approach, Stuart Russell and Peter Norvig](https://aima.cs.berkeley.edu/) <br>- [Reinforcement Learning: An Introduction, Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)<br>[CS285 at UC Berkeley, Deep Reinforcement Learning, Sergey Levine](https://rail.eecs.berkeley.edu/deeprlcourse/)" %}, but this isn't comprehensive. You should be familiar with common concepts and terminology in RL{% sidenote "2" "[Lilian Weng's list of RL notation is very useful here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#notations)" %}. For clarity, I'll try to spell out any jargon I use here.

## Recap

### Policy Gradient Methods
PPO is an on-policy reinforcement learning algorithm. It directly learns a stochastic policy function parameterised by $\theta$ representing the likelihood of action $a$ in state $s$, $\pi_{\theta}(a\vert s)$. Consider that we have some differentiable function, $J(\theta)$, which is a continuous performance measure of the policy $\pi_{\theta}$.  In the simplest case, we have  $J(\theta)={\mathbb{E}}\_{\tau\sim{\pi_{\theta}}}[R(\tau)]$, which is known as the [*return*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return){% sidenote "3" "The return represents the sum of rewards achieved over some time frame. This can be over a fixed timescale, i.e. the *finite-horizon* return, or over all time, i.e. the *infinite-horizon* return." %} over a [*trajectory*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#trajectories){% sidenote "99" "A trajectory, $\tau$, (also known as an episode or rollout) describes a sequence of interactions between the agent and the environment." %}, $\tau$. PPO is a kind of policy gradient method{% sidenote "98" "- [Policy Gradient Algorithms, Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-theorem)<br>- [Policy Gradients, CS285 UC Berkeley, Lecture 5, Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)" %} which directly optimizes the policy parameters $\theta$ against $J(\theta)$.

The [policy gradient theorem](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#proof-of-policy-gradient-theorem) gives, in an appropriate episodic setting, the REINFORCE-style estimator:
 
$$
\nabla_{\theta}J(\theta) = \mathbb{E}[\sum\limits_{t=0}^{\infty}\nabla_{\theta}\ln\pi_\theta(a_t|s_t)R_t]
$$

In other words, the gradient of our performance measure $J(\theta)$ with respect to our policy parameters $\theta$ points in the direction of increasing the expected return $J(\theta)$. Crucially, this shows that we can estimate the true gradient using an expectation of the sample gradient - the core idea behind the REINFORCE{% sidenote "4" "[Reinforcement Learning: An Introduction, 13.3 REINFORCE: Monte Carlo Policy Gradient, Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)" %} algorithm. This is great. The return in this expression can now be replaced with alternative estimators of the policy-gradient contribution. In particular, replacing the return with the advantage function preserves the expected gradient while typically reducing variance; practical advantage estimates may introduce some bias${% sidenote "5" "[Section 2, High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et. al. 2016](https://arxiv.org/pdf/1506.02438.pdf)" %} :

$$
\begin{aligned}
\nabla_{\theta}J(\theta) &= \mathbb{E}\left[\sum\limits_{t=0}^{\infty}\nabla_{\theta} \ln\pi_{\theta}(a_t|s_t) \Phi_t\right]&(1)
\end{aligned}
$$

Modern implementations of PPO express the policy-gradient contribution in terms of the [*advantage function*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions), and in practice use an estimate $\hat{A}_t$ of this quantity. This function measures the *advantage* of a particular action in a given state, i.e. how much better the expected return from taking this action is than the expected return from sampling an action according to the policy. Briefly described here, the advantage function takes the form

$$
A^{\pi}(s_t, a_t)= Q^\pi(s_t, a_t) - V^{\pi}(s_t)
$$

where $V(s)$ is the state-value function, and $Q(s, a)$ is the state-action value function, or Q-function{% sidenote "87" "The value function, $V(s)$, describes the expected return from starting in state $s$. Similarly the state-action value function, $Q(s, a)$, describes the expected return from starting in state $s$ and taking action $a$. See also [Reinforcement Learning: An Introduction, 3.7 Value Functions, Sutton and Barto](http://incompleteideas.net/book/ebook/node34.html)" %}.

I've found it easier to intuit the nuances of PPO by following the narrative around its motivations and predecessor. PPO builds on the same trust-region motivation as the [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (TRPO) method, which constrains policy changes during optimisation. The TRPO surrogate objective is defined as{% sidenote "6" "I'm omitting $ln$ from $ln\,\pi(a_{t}\vert s_{t})$ for brevity from here on." %} {% sidenote "7" "- [Proximal Policy Optimization Algorithms, Section 2.2, Schulman et al.](https://arxiv.org/abs/1707.06347)<br>- [Trust Region Policy Optimization, Schulman et al.](https://arxiv.org/pdf/1502.05477.pdf) for further details on the constraint."%} :

$$
\begin{aligned}
J(\theta) = \,&\mathbb{E} \left[\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
A_t\right]\\
\textrm{subject to}\,\, &\mathbb{E}[\textrm{KL}(\pi_{\theta_{\mathrm{old}}}\vert\vert\pi_\theta)] \le \delta
\end{aligned}
$$

Where KL is the Kullback–Leibler divergence (a measure of divergence between two probability distributions), and the ratio measures how much the new policy changes the probability of the sampled action relative to the policy that generated the trajectory:

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$$

Policy gradient methods optimise policies through iterative gradient updates to parameters $\theta$. The old policy, $\pi_{\theta_{\mathrm{old}}}(a_t|s_{t})$, is the one used to generate the current trajectory, and the new policy, $\pi_{\theta}(a_t|s_{t})$ is the policy currently being optimised{% sidenote "8" "Note: at the start of a series of policy update steps, we have $\pi_{\theta_{\mathrm{old}}}(a_t|s_{t})=\pi_{\theta}(a_t|s_{t})$, so $r_t(\theta)=1$." %}. For a positive advantage, the surrogate objective encourages increasing the probability of the sampled action relative to the old policy; for a negative advantage, it encourages decreasing it{% sidenote "9" "The surrogate gradient encourages $\pi_\theta(a_t|s_t)$ to increase relative to $\pi_{\theta_{\mathrm{old}}}(a_t|s_t)$ when $A_t>0$, and decrease when $A_t<0$." %}. The core principle of TRPO (and PPO) is to prevent excessively large policy changes during optimisation. Updating using the ratio between old and new policies in this way allows for selective reinforcement or penalisation of actions whilst grounding updates relative to the original, stable policy{% sidenote "10" "Consider optimising our policy using eqn. 1 - without normalising the update w.r.t. the old policy, updates to the policy can lead to catastrophically large updates." %}.

### PPO
PPO replaces TRPO's constrained optimisation with a clipped surrogate objective. PPO collects data using $\pi_{\theta_{\mathrm{old}}}$, then performs multiple optimisation epochs on that frozen batch; the likelihood ratio accounts for the changing policy, while clipping limits how far the surrogate can benefit from moving away from it:

$$
J(\theta)=
\mathbb{E}\left[
\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

To break this somewhat dense equation down, we can substitute our earlier expression $r_t(\theta)$ in:
 
 $$
 J(\theta)=\mathbb{E}[\textrm{min}(r_t(\theta)\,A_t, \textrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t)]
 $$
 
 The ratio is clipped inside the surrogate objective${% sidenote "11" "$\epsilon$ is usually set to $\sim0.1$." %}. For positive advantages, the objective stops benefiting from ratios above $1+\epsilon$; for negative advantages, it stops benefiting from ratios below $1-\epsilon$. This produces a conservative, pessimistic surrogate objective rather than a hard constraint on the actual policy ratio or a limit on the size of the parameter update.

So far, we've introduced the concept of policy gradients, objective functions in RL, and the core concepts behind PPO. Reinforcement learning algorithms place significant emphasis in reducing variance during optimisation{% sidenote "12" "Stochasticity in environment dynamics, delayed rewards, and exploration-exploitation tradeoffs all contribute to unstable training." %}. This becomes apparent when estimating the advantage function, which relies on sampled rewards obtained during a trajectory. Practical implementations of policy gradient algorithms reduce variance by also estimating an *on-policy* state-value function $V_\phi(s_t)$, which is the expected discounted return an agent receives from starting in state $s_t$, and following policy $\pi$ thereafter. Jointly learning a value function and policy function in this way is known as the Actor-Critic framework{% sidenote "13" "- [Reinforcement Learning: An Introduction, 6.6 Actor-Critic Methods, Sutton and Barto](http://incompleteideas.net/book/ebook/node66.html)<br>- [A (Long) Peek into Reinforcement Learning, Value Function, Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#value-function)" %}.  
### Value Learning and Actor-Critics
Value-function learning involves approximating the expected discounted return from following a policy from a current state. The value function is learned alongside the policy, and the simplest method is to minimise a mean-squared-error objective against the discounted return{% sidenote "14" "The *discounted return* down-weights rewards obtained in the future by an exponential discount factor $\gamma^l$, i.e. rewards in the distant future aren't worth as much as near-term rewards." %}, $G_t=\sum\limits_{l=0}^{\infty}\gamma^{l}r_{t+l}$:

$$
\phi^{*}=\operatorname*{arg\,min}_{\phi}\mathbb{E}_{\tau\sim\pi}[(V_\phi(s_t)-G_t)^2]
$$

We can now use this to learn a lower-variance estimator of the advantage function{% sidenote "15" "The on-policy Q-function is defined as the expected discounted return of taking action $a_t$ in state $s_t$, and following policy $\pi$ thereafter: $Q^\pi(s_t, a_t) = \mathbb{E}[r_t+\gamma V^\pi(s_{t+1})\mid s_t,a_t]$" %}. The true Q-function and advantage are:

$$
\begin{aligned}
Q^\pi(s_t, a_t) &= \mathbb{E}\left[r_t+\gamma V^\pi(s_{t+1}) \mid s_t, a_t\right]\\
A^\pi(s_t, a_t) &= Q^\pi(s_t, a_t)-V^\pi(s_t)\\
 &= \mathbb{E}\left[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t) \mid s_t,a_t\right]
\end{aligned}
$$

Because $V^\pi$ is unknown, we use the learned approximation $V_\phi$ to construct the practical estimator:

$$
\begin{aligned}
\hat{A}_t
&= r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\\
&= \delta_t^{V_\phi}
\end{aligned}
$$

If $V_\phi = V^\pi$, the temporal-difference{% sidenote "16" "[Reinforcement Learning: An Introduction, 6. Temporal-Difference Learning, Sutton and Barto](http://incompleteideas.net/book/ebook/node60.html)" %} residual, $\delta_t$, is an unbiased one-sample estimator of the advantage conditional on $(s_t, a_t)$. When $V_\phi$ is only an approximation, it is an approximate advantage estimator and can have approximation error. The same TD residual also forms the basis of semi-gradient TD(0) value learning:

$$
\begin{aligned}
\phi&\leftarrow \phi+ \alpha(r_t+\gamma V_\phi(s_{t+1}) - V_\phi(s_t))\nabla_\phi V_\phi(s_t)\\
\phi&\leftarrow \phi + \alpha\,\delta_t\nabla_\phi V_\phi(s_t)\\
\textrm{where}\,\,\delta_t&=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
\end{aligned}
$$

### Generalised Advantage Estimation (GAE)

There's one more thing we can do to reduce variance. The current advantage estimator relies heavily on bootstrapping from the learned value function, which gives low variance but can be sensitive to value-function approximation. Sampling more actual rewards gives a less heavily bootstrapped estimate, but can have higher variance. This is the central idea behind *n*-step returns{% sidenote "17" "- [DeepMimic Supplementary A, Peng et al.](https://arxiv.org/pdf/1804.02717.pdf)<br>- [Reinforcement Learning: An Introduction, 7.1 n-step TD Prediction, Sutton and Barto](http://incompleteideas.net/book/ebook/node73.html)" %}: we trade off reliance on the learned value function against the variance of Monte Carlo-style reward sampling.

Consider the term $\delta_t$ in our estimation of the advantage function. We take the initial reward observed from the environment, $r_t$, then bootstrap future estimated discounted rewards, and subtract a baseline estimated value function for the state{% sidenote "18" "Daniel Takeshi's [post](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) on using baselines to reduce variance of gradient estimates is useful here." %}. For a single time-step, this can be denoted as:

$$
\hat{A_{t}}^{(1)} = r_t+\gamma V_{\phi}(s_{t+1})- V_{\phi}(s_t) = \delta_t^{V_\phi}
$$

What if we sample rewards from multiple time-steps, and then estimate the future discounted rewards from then on? Let's denote $\hat{A_t}^{(k)}$ as follows:

$$
\begin{aligned}
\hat{A_{t}}^{(1)} &= r_t+\gamma V_{\phi}(s_{t+1})- V_{\phi}(s_t)&&=\delta_t^{V_\phi}\\

\hat{A_{t}}^{(2)} &= r_{t}+\gamma r_{t+1} + \gamma^{2}V_{\phi}(s_{t+2})- V_{\phi}(s_t) &&= \delta_t^{V_{\phi}}+ \gamma\delta_{t+1}^{V_\phi}\\

\hat{A_{t}}^{(k)} &= \sum\limits_{l=0}^{k-1}\gamma^l r_{t+l} + \gamma^{k}V_{\phi}(s_{t+k})- V_{\phi}(s_t)&&= \sum\limits_{l=0}^{k-1}\gamma^l\delta_{t+l}^{V_\phi}\\

\hat{A_{t}}^{(\infty)} &= \sum\limits_{l=0}^{\infty}\gamma^l r_{t+l} - V_{\phi}(s_t)&&= G_t-V_{\phi}(s_t)\\
\end{aligned}
$$

$n=1$ and $n=\infty$ now become extreme cases in $n$-step return estimation. Observe that for $k=\infty$ we recover the infinite-horizon discounted return minus our baseline estimated value function. With the true value function, an $n$-step return has the correct expected value; with an approximate value function, bootstrapping introduces approximation bias. GAE{% sidenote "19" "- [High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et. al. 2016](https://arxiv.org/pdf/1506.02438.pdf)<br>- [Notes on the Generalized Advantage Estimation Paper, Daniel Takeshi](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)" %} introduces a parameter, $\lambda \in[0,1]$, to take an exponentially weighted average over every $k$-th step estimator. Using notation from the paper{% sidenote "20" "The identity $\frac{1}{1-\lambda} = 1 + \lambda + \lambda^2+\lambda^3 + ...$ for $|\lambda|<1$ is useful here."%}, we can derive a *generalized advantage estimator* for cases where $0<\lambda<1$:

$$
\begin{aligned}
\hat{A_{t}}^{\gamma \lambda} &= (1-\lambda)(\hat{A_{t}}^{(1)} + \lambda\hat{A_{t}}^{(2)} + \lambda^2\hat{A_{t}}^{(3)} + ...)\\
&= (1-\lambda)(\delta_{t}^{V_\phi} + \lambda(\delta_{t}^{V_{\phi}}+ \gamma\delta_{t+1}^{V_\phi})  + \lambda^{2}(\delta_{t}^{V_\phi}+\gamma\delta_{t+1}^{V_{\phi}}+ \gamma^2\delta_{t+2}^{V_\phi})+ ...)\\
&= (1-\lambda)(\delta_{t}^{V_\phi}(1+\lambda+\lambda^2+...) + \gamma\delta_{t+1}^{V_\phi}(\lambda+\lambda^2+\lambda^3+...)+...)\\
&= (1-\lambda)\left(\delta_{t}^{V_\phi}\left(\frac{1}{1-\lambda}\right)+\gamma\delta_{t+1}^{V_\phi}\left(\frac{\lambda}{1-\lambda}\right)+\gamma^2\delta_{t+2}^{V_\phi}\left(\frac{\lambda^2}{1-\lambda}\right)+...\right)\\
&= \sum\limits_{l=0}^{\infty}(\gamma \lambda)^{l}\delta_{t+l}^{V_\phi}
\end{aligned}
$$

Thus:

$$
\hat{A}_{t}^{\mathrm{GAE}(\gamma,\lambda)}
= \sum\limits_{l=0}^{\infty}(\gamma \lambda)^{l}\delta_{t+l}^{V_\phi}
$$

Great. As you may have noticed, there's two special cases for this expression - $\lambda=0$, and $\lambda=1$: 

$$
\begin{aligned}
\hat{A_t}^{\gamma*0}&=\hat{A_t}^{(1)}=r_t+\gamma V_{\phi}(s_{t+1})- V_{\phi}(s_t)=\delta_t\\
\hat{A_t}^{\gamma*1}&= \sum\limits_{l=0}^{\infty}\gamma^{l}\delta_{t+l}^{V_\phi}\\
&=(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_{t})) \\
&+ \gamma(r_{t+1}+\gamma V_\phi(s_{t+2})-V_\phi(s_{t+1}))\\
&+ \gamma^2(r_{t+2}+\gamma V_\phi(s_{t+3})-V_\phi(s_{t+2}))+ ... \\
&= r_t \bcancel{\gamma V_\phi(s_{t+1})}-V_\phi(s_{t})\\
&+ \gamma r_{t+1} + \bcancel{\gamma^{2}V_\phi(s_{t+2})}-\bcancel{\gamma V_\phi(s_{t+1})}\\
&+ \gamma^2 r_{t+2} + \gamma^{3}V_\phi(s_{t+3})-\bcancel{\gamma^2 V_\phi(s_{t+2})}+...\\
&=\sum\limits_{l=0}^{\infty}\gamma^{l}r_{t+l}- V_{\phi}(s_t)=G_t-V_\phi(s_t)
\end{aligned}
$$

We see that $\lambda=0$ gives the one-step TD residual, corresponding to the most heavily bootstrapped GAE estimator. For $\lambda=1$, we obtain a high-variance advantage estimator, which is simply the discounted return minus our baseline estimated value function. For $\lambda \in (0,1)$, we obtain an advantage estimator which allows control of the bias-variance tradeoff. The authors note that the two parameters, $\gamma$ and $\lambda$, control variance in different ways. $\gamma$ determines the discounted objective and how strongly future rewards are discounted, while $\lambda$ controls how much bootstrapping and multi-step information are used in the estimator. GAE therefore does not change the PPO surrogate objective itself; it changes the advantage estimates supplied to that objective.
### Pseudocode

Tying everything together, we can show the general process of updating our policy and value function using PPO for a single trajectory of fixed length $N$:<br><br>
> Given policy estimator $\pi_\theta$, value function estimator $V_\phi$, $\gamma$, $\lambda$, $N$ time-steps.
>>For $t=1,2,...,N:$
>>
>>>Run policy $\pi_\theta$ in environment and collect rewards, observations, actions, and value function estimates $r_{t}, s_{t}, a_t, v_t$ where $v_t=V_\phi(s_t)$.
>>
>>Evaluate $v_{N+1}=V_\phi(s_{N+1})$.
>>
>>Compute $\delta_t^{V_\phi}=r_t+\gamma m_t v_{t+1}-v_t$ for $t=N,...,1$, where $m_t=0$ if transition $t$ terminates the episode and $m_t=1$ otherwise.
>>
>>Set $\hat{A}_{N+1}=0$.
>>
>>Compute GAE backwards, $\hat{A_{t}}=\delta_t+\gamma\lambda m_t\hat{A}_{t+1}$ for $t=N,...,1$.
>>
>>Compute $\pi_\theta(a_t|s_t)$ $\log$-probabilities for the stored actions and states.
>>
>>Optimise $\theta$ using $J(\theta)$, the PPO objective{% sidenote "21" "This is usually done over $M$ minibatch steps for $M \le N$. $\pi_{\theta_{\mathrm{old}}}$ is fixed as the initial policy at the start of the trajectory, and $\pi_\theta$ is taken as the policy at the current optimisation step." %}. 
>>
>> Compute a return target $\hat{R}_t$, commonly $\hat{A}_t+V_{\phi_{\mathrm{old}}}(s_t)$, and optimise $\phi$ using $L_V=(V_\phi(s_t)-\hat{R}_t)^2$ using stored data${% sidenote "22" "Similarly to the policy optimisation step, this is also done over $M$ steps. $V_\phi$ is taken as the value function at the current optimisation step, and the return target can correspond to a bootstrapped $n$-step return." %}.
>
>... repeat!


Feedback and corrections are very welcome. I'll follow this post with implementation details soon. For now, I'd recommend the excellent resource by [Shengyi Huang](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) on reproducing PPO implementations from popular RL libraries. If you're able to implement the policy update from the PPO paper, hopefully there's enough detail here that you're able to reproduce other implementations of PPO. Thank you so much for reading.  
