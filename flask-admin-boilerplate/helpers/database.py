from pymongo import MongoClient
import configuration

connection_params = configuration.connection_params

#connect to mongodb
mongoconnection = MongoClient(
    "mongodb+srv://hackathon-ram:Hellotest@cluster0.zfh3k.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
)


db = mongoconnection.databasename
