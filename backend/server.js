const express = require('express');
const cors = require("cors");
const session = require("express-session")
const { MongoStore } = require("connect-mongo");
require('dotenv').config();

const backend_routes = require('./routes/index');

const connectDB = require('./config/db');

const app = express();

const port = process.env.PORT || 5000;

connectDB();
app.use(cors({
     origin: "http://localhost:5173",
     credentials: true
}));
app.use(express.json());

app.use(session({
     secret: process.env.JWT_KEY || "your-secret-key",

     resave: false,
     saveUninitialized: false,
     store: MongoStore.create({
          mongoUrl: process.env.MONGO_URI || "mongodb://127.0.0.1:27017/admin_sessions",
          ttl: 24 * 60 * 60 // 1 day
     }),
     cookie: {
          secure: false, // set true if HTTPS
          httpOnly: true,
          maxAge: 24 * 60 * 60 * 1000 // 1 day
     }
}));

app.use('/api/backend', backend_routes)

app.listen(port, () => {
     console.log(`Server running at http://localhost:${port}`);
})