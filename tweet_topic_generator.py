import pandas as pd
import re
from datetime import datetime

class TweetTopicGenerator:
    def __init__(self, client, batch_size=500, num_batches=5):
        self.client = client
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.topics = {}
        self.max_id = 0

    def generate_prompt(self, tweets, current_batch):
        system_prompt = f"""
            You are an AI assistant tasked with identifying topics from a collection of tweets.
            These tweets are short tweets from Twitter users, which may include various forms of content
            such as news, opinions, discussions, events, especially war-events and sometimes spam or irrelevant information. Your job is to
            analyze the tweets and generate a list of coherent topics that represent the main subjects discussed.
            If a topic that represents the main subjects discussed already exists, don't add it again.
            Focus on identifying meaningful and relevant topics while ignoring spam or unrelated content.
            Be careful not to add topics that are similar to the ones already listed.

            You are currently processing batch {current_batch+1} of {self.num_batches}.

            When providing your response, you should categorize your decisions into two types:
            - add: When you identify a new topic not related to any existing topic (no ID needed).
            - remove: When a topic is no longer relevant or needs to be deleted because you added a more refined one (provide the topic ID).

            You have to respond in JSON format and in Hebrew language.

            Here are the current topics with their IDs:
        """

        current_topics = "\n".join([f"{topic}: {topic_id}" for topic, topic_id in self.topics.items()])
        user_prompt = f"Here is a batch of tweets:\n{''.join(map(str, tweets))}\n\nGenerate a list of topics from these tweets and specify your actions for each topic. Make sure to avoid adding duplicate or similar topics to the existing list:\n"

        example_response = (
            "\nExample JSON response:\n"
            "{\n"
            '    "add": [\n'
            '        "Judicial Reform in Israel",\n'
            '        "Public Sentiment on Military Actions",\n'
            '        "Humanitarian Issues in Gaza",\n'
            '        "Impact of Supreme Court Decisions on Legislation",\n'
            '        "Public Protests Against Government Policies",\n'
            '        "Political Accountability and Corruption Allegations",\n'
            '        "Civil-Military Relations in Israel",\n'
            '        "International Response to the Gaza Conflict",\n'
            '        "Media Coverage of War Events",\n'
            '        "Public Opinion on Humanitarian Aid to Gaza"\n'
            '    ],\n'
            '    "remove": [\n'
            '        {"topic": "Judicial Decisions and Implications", "id": 12},\n'
            '        {"topic": "War and Conflict in Gaza", "id": 9},\n'
            '        {"topic": "Political Accountability", "id": 11},\n'
            '        {"topic": "Public Protests and Demonstrations", "id": 20}\n'
            '    ]\n'
            "}\n"
            "Make sure you not only adding new topics! You must response the topics names in Hebrew"
        )

        messages = [
            {"role": "system", "content": system_prompt + current_topics + example_response},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def process_tweets(self, df):
        for current_batch in range(self.num_batches):
            batch = df.sample(n=self.batch_size)['Lemmatized Cleaned Text'].tolist()
            messages = self.generate_prompt(batch, current_batch)
            topics = self.client.llm_call(messages, max_tokens=1000, json_format=True)
            print(topics)

            self.update_topics(topics)

        return self.get_topics_with_ids()

    def update_topics(self, response):
        try:
            topics = eval(response)
            new_topics = topics.get("add", [])
            remove_topics = topics.get("remove", [])

            for topic in new_topics:
                if topic not in self.topics:
                    self.max_id += 1
                    self.topics[topic] = self.max_id

            for topic_info in remove_topics:
                topic = topic_info["topic"]
                if topic in self.topics:
                    del self.topics[topic]

        except (SyntaxError, KeyError, TypeError) as e:
            print(f"Error processing response: {e}")

    def get_topics_with_ids(self):
        # Note that we are not using this func, it is used for the dataset creation process
        return [{"topic": topic, "id": topic_id} for topic, topic_id in self.topics.items()]
    
    def finalize_topics(self):
        topics_list = [topic for topic in self.topics.keys()]
        topics_string = "\n".join(topics_list)

        system_prompt = """
        You are a helpful assistant. The user will provide you a list of topics. Each topic is in a new line.
        Your task is to provide a JSON of 20 topics that represents well all of the topics.
        Make sure the subjects are logical and distinct from one another to avoid duplication and ensure variety.
        
        Whenever encountering abbreviated words, replace them with their full forms. For instance, the abbreviation בג 
        should be replaced with בגץ. This rule should be applied universally to similar abbreviations.

        Examples:
        בג -> בגץ
        "ביקורת על בג" should be "ביקורת על בגץ"
        "ההשלכות של החלטות בג" should be "ההשלכות של החלטות בגץ"
        "הפגנות נגד בג" should be "הפגנות נגד בגץ"


        You must response the topics names in Hebrew

        Example JSON response:
        {
            "response": [
                "topic num 1",
                "topic num 2",
                ...
                "topic num 20"
            ]
        }
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topics_string},
        ]
        response = self.client.llm_call(messages, max_tokens=4000, model="gpt-4o", json_format=True)
        return response
    
# # Example usage of `TweetTopicGenerator`
# df = pd.read_csv('lemmatized_tweets.csv')

# client = OpenAIGPTClient()
# topic_generator = TweetTopicGenerator(client, batch_size=500, num_batches=3)

# topics = topic_generator.process_tweets(df)
# print(topics)

# final_topics = topic_generator.finalize_topics()
# print(final_topics)