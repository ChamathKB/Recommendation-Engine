from configparser import ConfigParser
from confluent_kafka import Consumer, OFFSET_BEGINNING

CONFIG_FILE = 'config.ini'

class event_consumer():

    def __init__(self, topic):
        self.topic = topic

    def load_config(CONFIG_FILE):
        config_parser = ConfigParser()
        config_parser.read_file(CONFIG_FILE)
        config = dict(config_parser['default'])
        config.update(config_parser['consumer'])
        return config


    def reset_offset(consumer, partitions, load_config):
        consumer = Consumer(load_config)
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)

    
    def run_consumer(self, reset_offset, consumer):

        topic = self.topic
        consumer.subscribe([topic], on_assign=reset_offset)

        try:
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    print("Waiting...")
                elif msg.error():
                    print("ERROR: %s".format(msg.error()))
                else:

                    print("Consumed event from topic {topic}: key = {key:12} value = {value:12}".format(
                        topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()