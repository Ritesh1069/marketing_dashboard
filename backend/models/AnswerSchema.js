const mongoose = require('mongoose');

const AnswerSchema = new mongoose.Schema({

    // id:{
    //     // required: true, 
    //     unique: true,
    //     type: String
    // },

    chp_no:{
       type:Number,
       required: true
    },
    
    ques_no:{
        type:Number,
        required: true
     },

    module_name:{
        type:String,
        required: true,
    },

    question:{
        type: String,
        required:true,
        unique:true
    },

    answer:{
        type: String,
        required:true
    }
    
})

const Answer = mongoose.model('Answer', AnswerSchema);
module.exports = Answer