const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 5000;

// Enable CORS
app.use(cors());

// Middleware to parse JSON requests
app.use(express.json());

// Endpoint to handle question queries
app.post('/ask', (req, res) => {
    const question = req.body.question;

    // Read the JSON file
    const jsonData = JSON.parse(fs.readFileSync(path.join(__dirname, 'final_response_log.json'), 'utf-8'));

    // Find the answer to the question
    const response = jsonData.find(item => item.question.toLowerCase() === question.toLowerCase());

    if (response) {
        res.json(response);
    } else {
        res.status(404).send('Question not found');
    }
});

// Serve static files from the client/build folder (React app)
app.use(express.static(path.join(__dirname, '../client/build')));

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
