const express = require('express');
const upload = require('../middleware/upload');
const { uploadVideo } = require('../controllers/upload_controller');

const router = express.Router();

router.post('/upload-video', upload.single('video'), uploadVideo);

module.exports = router;
