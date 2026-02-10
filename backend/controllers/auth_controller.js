const { getDb } = require('../config/db');
const bcrypt = require('bcrypt');

const signup = async (req, res) => {

    try {

        const { name, phone_no, email, password } = req.body;

        if (!name || !phone_no || !email || !password) {
            return res.status(400).json({ message: "All fields are required" })
        }

        if (password.length < 6) {
            return res.status(400).json({ message: "Password must be at least 6 characters long" });
        }

        const normalizedEmail = email.toLowerCase();

        const db = getDb();

        const existingUser = await db.collection('users').findOne({ email: normalizedEmail });

        if (existingUser) {
            return res.status(400).json({ message: "User already exists" })
        }

        const hasehpassword = await bcrypt.hash(password, 10);

        const newUser = {
            name,
            phone_no,
            email: normalizedEmail,
            password: hasehpassword,
            createdAt: new Date()
        };

        await db.collection('users').insertOne(newUser);

        return res.status(201).json({ message: "User registered successfully" })

    } catch (error) {
        console.error(error);
        return res.status(500).json({ message: "Internal server error" })
    }
}

const login = async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ message: "Email and password are required" });
        }

        const db = getDb();
        const normalizedEmail = email.toLowerCase();

        const user = await db.collection('users').findOne({ email: normalizedEmail });

        if (!user) {
            return res.status(404).json({ message: "User not found. Please signup to continue." });
        }

        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.status(401).json({ message: "Invalid email or password" });
        }

        req.session.user = {
            id: user._id,
            name: user.name,
            email: user.email,
            phone_no: user.phone_no
        };

        return res.status(200).json({
            message: "Login successful",
            user: req.session.user
        });

    } catch (error) {
        console.error(error);
        return res.status(500).json({ message: "Internal server error" });
    }
};

module.exports = { signup, login }