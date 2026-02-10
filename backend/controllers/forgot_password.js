const { getDb } = require("../config/db");
const crypto = require("crypto");
const { sendOtpEmail } = require("../services/send_email.service");
const bcrypt = require("bcryptjs");
require("dotenv").config();

const forgotpassword = async (req, res) => {
  try {
    const db = getDb();
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ message: "Email is required" });
    }

    const normalizedEmail = email.toLowerCase();

    const user = await db
      .collection("staff")
      .findOne({ email: normalizedEmail });

    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    const otp = crypto.randomInt(1000, 10000).toString();

    const hashedOtp = crypto
      .createHmac("sha256", process.env.OTP_SECRET)
      .update(otp)
      .digest("hex");

    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000);

    await db.collection("staff").updateOne(
      { email: normalizedEmail },
      {
        $set: {
          resetOtp: hashedOtp,
          resetOtpExpiry: otpExpiry,
        },
      },
    );

    await sendOtpEmail({
      to: normalizedEmail,
      otp,
    });

    res.status(200).json({
      message: "OTP sent to email successfully",
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      message: "Internal Server error",
    });
  }
};

const otpValidation = async (req, res) => {
  try {
    const { email, otp } = req.body;

    if (!email || !otp) {
      return res.status(400).json({
        message: "Email and otp is required",
      });
    }

    const normalizedEmail = email.toLowerCase();

    const hashedOtp = crypto
      .createHmac("sha256", process.env.OTP_SECRET)
      .update(otp)
      .digest("hex");

    const user = await db.collection("staff").findOne({
      email: normalizedEmail,
      resetOtp: hashedOtp,
      resetOtpExpiry: { $gt: new Date() },
    });

    if (!user) {
      return res.status(400).json({
        message: "Invalid or expired OTP",
      });
    }

    await db.collection("staff").updateOne(
      { email: normalizedEmail },
      {
        $unset: {
          resetOtp: "",
          resetOtpExpiry: "",
        },
      },
    );

    res.status(200).json({
      message: "Otp is validated",
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Server error" });
  }
};

const resetPassword = async (req, res) => {
  try {
    const db = getDb();
    const { email, newPassword, confirmPassword } = req.body;

    if (!email || !newPassword || !confirmPassword) {
      return res.status(400).json({
        message: "Email and  password are required",
      });
    }

    if (newPassword !== confirmPassword) {
      return res.status(400).json({
        message: "Passwords do not match",
      });
    }

    if (newPassword.length < 8 && newPassword.length > 12) {
      return res.status(400).json({
        message: "Password must be at least 8 characters long",
      });
    }

    const normalizedEmail = email.toLowerCase();

    const user = await db.collection("staff").findOne({
      email: normalizedEmail,
    });

    if (!user) {
      return res.status(400).json({
        message: "Invalid Email",
      });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(newPassword, salt);

    await db.collection("staff").updateOne(
      { email: normalizedEmail },
      {
        $set: {
          password: hashedPassword,
        },
      },
    );

    res.status(200).json({
      message: "Password reset successful",
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      message: "Server error",
    });
  }
};

module.exports = {
  forgotpassword,
  resetPassword,
  otpValidation,
};
