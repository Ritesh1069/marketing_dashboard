const express = require('express');
const db = require('./db');
const app = express();
const port = process.env.PORT || 4400;
const bodyParser = require('body-parser');
const Answer = require('./models/AnswerSchema');

app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('<h1>AI Tutor</h1>');
});

app.get('/answer', async (req, res) => {
    try {
        const data = await Answer.find();
        res.status(200).json({ data });
    } catch (error) {
        console.log(error);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.post('/answer', async (req, res) => {
    const data = req.body;
    const newAns = new Answer(data);

    try {
        const response = await newAns.save();
        console.log("Data saved successfully");
        res.status(200).json({ success: true });
    } catch (error) {
        console.log(error);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.delete('/answer/:id', async (req, res) => {
    const id = req.params.id;
    try {
        const response = await Answer.findByIdAndDelete(id);
        if (!response) {
            res.status(404).json({ error: "No element found" });
        } else {
            res.status(200).json({ success: "Data deleted successfully" });
        }
    } catch (error) {
        console.log("Error occurred while deleting:", error);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.listen(port, () => {
    console.log(`App listening on port ${port}`);
});

