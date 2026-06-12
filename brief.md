### Introduction
The Jump Trading Probability Cup is a forecasting competition hosted on the SportsPredict.com platform and sponsored by Jump Trading Group.

Participants are ranked based on the accuracy of their forecasts throughout the competition. 

Example questions (actual questions):
* At halftime, will Czechia have more corner kicks than South Korea?
* Will South Korea be caught offside 2 or more times?
* Will there be 2 or more total cards shown in the second half?
* Will both teams score AND the match have 3 or more total goals?
* Will a penalty kick be awarded in the match?
* Will South Korea have more shots on target than Czechia in the second half?
* Will South Korea win the match?
* Will South Korea score in the second half?
* Will Son Heung-min score a goal (excluding own goals)?
* Will Patrik Schick have at least 1 shot on target?

These are the archetypes of questions (can differ per match but in terms of outcome it's tracking, I suspect it will be in the same category)

### Scoring

The scoring system uses a relative Brier score

**What is a Brier Score and why should I care?**

The Brier Score is how we measure the accuracy of your predictions. 

Here's the simple version: you give a prediction between 0 and 100. That number gets converted to a decimal (so 75 becomes 0.75). After the match, the outcome is either YES (1.0) or NO (0.0). Your Brier Score is the squared distance between your prediction and the outcome.

Example:

Your prediction 
75 = 0.75

Outcome
YES = 1.0

Distance
1.0 − 0.75 = 0.25

Brier Score
0.25² = 0.0625

The key thing to remember: the Brier Score measures error, so lower is better. A score of 0 means you were perfect. A score of 1 means you were as wrong as possible.

You don't earn points from your raw Brier Score. Instead, you earn Relative Brier Points by being more accurate than the other participants. Here's how it works:

After each question, we calculate the field average Brier Score — the average of everyone's Brier Scores. Then we subtract your Brier Score from that average. If your score is lower (better) than the average, the difference is positive and you earn points. If your score is higher (worse), you lose points.

Field Average Brier
0.450
Your Brier Score
0.202
Your Edge
0.450 − 0.202 = +0.248
Relative Brier Points
0.248 × 100 = +24.8
Your cumulative total of these points across all questions determines your rank on the leaderboard.

### Strategy 
**"I think Brazil will beat Costa Rica. Why shouldn't I just predict 100?"**

Short answer: because if you're wrong, the penalty is catastrophic, and even if you're right, you barely benefit versus predicting, say, 90.

Let's do the math. Say you predict 100 (= 1.0):

If YES (Brazil wins)
Distance 1.0 − 1.0 = 0.0 → Brier = 0 (perfect)
If NO (upset)
Distance 1.0 − 0.0 = 1.0 → Brier = 1.0 (worst possible)
Now compare that to predicting 90 (= 0.90):

If YES
Distance 1.0 − 0.90 = 0.10 → Brier = 0.01
If NO
Distance 0.90 − 0.0 = 0.90 → Brier = 0.81
By predicting 90 instead of 100, you give up almost nothing when you're right (0.01 vs 0.00), but you save yourself significantly when you're wrong (0.81 vs 1.00). The scoring system mathematically rewards you for expressing your actual confidence level, not for exaggerating it. This is called being 'well-calibrated.'

The bottom line: predict 100 only if you are literally certain — like, the match has already been played and you know the result. Otherwise, always leave yourself some room.

**Can someone game the system?**
No. The Brier Score is what mathematicians call a 'proper scoring rule.' This is a technical term that means the scoring system is specifically designed so that your best strategy is always to report your true belief. Any attempt to be clever — predicting more extreme numbers to 'win bigger,' hedging toward 50 to play it safe, or copying other people's predictions — will, on average, give you a worse score than just being honest.

Here's why copying doesn't work: if you match the field average on every question, your edge is always zero. You'd never earn a single point. You have to be different from the crowd AND more accurate to gain an advantage.

And hedging toward 50 on everything is equally bad — you'd be saying 'I have no idea' on every question, and anyone with actual knowledge will consistently outperform you.

The only winning strategy is genuine accuracy. Do your research, form an honest opinion, and submit it.

**Should I answer as many questions as possible, or focus on the ones I'm most confident about?**

This is one of the most important strategic decisions in the contest, and the answer is nuanced.

The case for being selective: if you only answer questions where you have a genuine edge (strong knowledge, good research, a clear opinion), your average points per question will be high. You avoid the risk of losing points on questions where you'd just be guessing.

The case for answering everything: every question is an opportunity to earn points. If you skip a question where you would have beaten the field average, that's points left on the table. And more questions means more chances for your skill to show through — one bad prediction gets diluted across many.

Here's a practical rule of thumb: if you have any opinion at all, submit it. The field average often reflects a sort of consensus — if you have information or insight that others don't, even a slightly better prediction earns you points. The only time to skip a question is if you truly have zero knowledge and would literally be guessing at random.

Remember: the leaderboard ranks by cumulative points, not average points. A player with +20 points from 30 questions beats a player with +18 points from 5 questions, even though the second player has a higher average.


**What happens if I just predict 50 on everything?**
Predicting 50 on everything is like saying 'I have absolutely no idea' for every question. It's the safest possible strategy — but it won't win you the contest.

If the field average is close to 50, your edge will be near zero on every question. You'll hover around the middle of the leaderboard, never gaining or losing much. But if the field is generally better calibrated than a coin flip (and they likely will be), you'll slowly bleed points.

You predict 50 (= 0.50), outcome YES
Brier = 0.50² = 0.25
Someone predicts 70 (= 0.70), outcome YES
Brier = 0.30² = 0.09
They beat you by a wide margin.

The 50-everywhere strategy guarantees mediocrity. To win, you need to have conviction when you have information.


### Resources
Team news
The Athletic · BBC Sport · ESPN FC
Weather
AccuWeather · Windy · venue forecasts
Betting markets
Pinnacle · Betfair Exchange · OddsPortal
Statistical models
FiveThirtyEight · Opta · FBref · Understat
AI tools
ChatGPT · Claude · Gemini · Perplexity
Expert analysis
Tifo Football · Tactical Times · podcasts


### Bots
I am allowed to create 2 bots that can submit predictions on my behalf using the API (documentation at https://sportspredict.com/probabilitycup/api). These bots submit independent predictions to my account, and thus should be used to maximise the chances.