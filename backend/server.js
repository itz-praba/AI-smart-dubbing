const express = require('express');

require('dotenv').config();

const backend_routes = require('./routes/index');

const connectDB = require('./config/db');

const app = express();

const port = process.env.PORT || 5000;

connectDB();

app.use('/api/backend',backend_routes)

app.listen(port, () => {
     console.log(`Server running at http://localhost:${port}`);
})