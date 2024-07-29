import pandas as pd
import json
# Importing the pandas library for data manipulation and analysis.

from openai_gpt_client import OpenAIGPTClient
# Importing the OpenAIGPTClient class, which is assumed to be a client for interacting with OpenAI's GPT models.

from tweet_topic_generator import TweetTopicGenerator
# Importing the TweetTopicGenerator class, which handles the generation and processing of topics from tweets.

print("📄 Reading the CSV file into a DataFrame...")
df = pd.read_csv('lemmatized_tweets.csv')
# Reading a CSV file named 'lemmatized_tweets.csv' into a pandas DataFrame. 
# This DataFrame contains tweets that have been cleaned and lemmatized in the Jupyter notebook.

print("🔄 Initializing the OpenAIGPTClient...")
client = OpenAIGPTClient()
# Creating an instance of the OpenAIGPTClient class. This client will be used to interact with the OpenAI API for processing tweets.

print("🧠 Creating the TweetTopicGenerator instance...")
topic_generator = TweetTopicGenerator(client, batch_size=500, num_batches=3)
# Creating an instance of the TweetTopicGenerator class with the client, 
# setting the batch size to 500 tweets per batch, and specifying that 3 batches will be processed.
# `batch_size` and `num_batches` are hyperparameters we can tune based on the need.

print("💬 Processing tweets to generate initial topics...")
topics = topic_generator.process_tweets(df)
# Calling the process_tweets method of the TweetTopicGenerator instance. 
# This method processes the tweets in the DataFrame and generates a list of topics. 
# The resulting topics are stored in the 'topics' variable.

print("\n✅ Initial topics generated:")
print(topics)
# Printing the list of generated topics to the console.

print("\n🔍 Finalizing the topics into a refined list of 20 topics...")
response = topic_generator.finalize_topics()
# Calling the finalize_topics method of the TweetTopicGenerator instance. 
# This method refines the list of topics to produce a final list of 20 topics. 
# The resulting refined topics are stored in the 'final_topics' variable.

print("\n🎯 Final list of refined topics:")
final_topics = json.loads(response)['response']
print(final_topics)
# Printing the final list of refined topics to the console.

# 🤩 Example list of topics: 
# 'מצב הביטחון בצפון ישראל'
# 'ביקורת על ממשלת ישראל'
# 'התגובות למלחמה בעזה'
# 'ההשפעה של בגץ'
# 'המשפט על דמוקרטיה בישראל'
# 'ההתמודדות עם חטופים בעזה'
# 'המצב הכלכלי בישראל בעקבות המלחמה'
# 'הפגנות נגד ממשלת ישראל'
# 'היחסים עם חמאס'
# 'ההשפעה של מלחמת אוקטובר על החברה הישראלית'
# 'הבעיות במערכת המשפט בישראל'
# 'השפעת המלחמה על החברה הישראלית'
# 'המצב הנפשי של חיילים במלחמה'
# 'הביקורת על בגץ'
# 'ההפגנות נגד ממשלת ישראל בעקבות המלחמה'
# 'המצב הכלכלי בעקבות המלחמה'
# 'היחסים עם חמאס וההתקפות על ישראל'
# 'ההשפעה של חמאס על ביטחון ישראל'
# 'המצב הכלכלי בעקבות התייקרות מחירי הדלק'
# 'המאבק המשפטי בישראל'