const express = require('express');
const { signup, login } = require('../controllers/auth_controller');
const { forgotpassword, otpValidation, resetPassword } = require('../controllers/forgot_password');

const router = express.Router();

router.post('/signup', signup);

router.post('/login',login);

router.post('/forgot-password',forgotpassword);

router.post('/otp-validation',otpValidation);

router.post('/reset-password',resetPassword);

module.exports = router;