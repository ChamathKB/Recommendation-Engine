from random import choice
from configparser import ConfigParser
from confluent_kafka import Producer

CONFIG_FILE = 'config.ini'

class event_producer():

    def __init__(self, topic):
        self.topic = topic

    def load_config(CONFIG_FILE):
        config_parser = ConfigParser()
        config_parser.read_file(CONFIG_FILE)
        config = dict(config_parser['default'])
        return config


    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))


    def run_producer(self, load_config, delivery_callback, user_id):
        producer = Producer(load_config)
        topic = self.topic
        count = 0
        for _ in range(10):

            user_id = choice(user_id)
            producer.produce(topic, user_id, callback=delivery_callback)
            count += 1

        producer.poll(10000)
        producer.flush()