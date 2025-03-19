const mongoose = require('mongoose')
require('dotenv').config()
const Mongo_URL = process.env.MONGO_URL

mongoose.connect(Mongo_URL)

const db = mongoose.connection;

db.on('connected', ()=>{
    console.log("connected to mongoDB server")
})

db.on('error', ()=>{
    console.log("error occoured while connecting to mongo DB")
})

db.on('disconnected', ()=>{
    console.log("dissconnectd to mongoDB server")
})

module.exports = db

