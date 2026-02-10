const { Upload } = require('@aws-sdk/lib-storage');
const { v4: uuidv4 } = require('uuid');
const s3 = require('../config/s3');

const uploadVideo = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'Video file is required' });
        }

        const videoKey = `videos/${uuidv4()}-${req.file.originalname}`;

        // Multipart upload (STREAMING)
        const upload = new Upload({
            client: s3,
            params: {
                Bucket: process.env.AWS_S3_BUCKET_NAME,
                Key: videoKey,
                Body: req.file.stream, // 🔥 STREAM
                ContentType: req.file.mimetype
            },
            queueSize: 4,          // parallel uploads
            partSize: 10 * 1024 * 1024, // 10MB chunks
            leavePartsOnError: false
        });

        await upload.done();

        const videoUrl = `https://${process.env.AWS_S3_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${videoKey}`;

        return res.status(201).json({
            message: 'Video uploaded successfully',
            videoUrl
        });

    } catch (error) {
        console.error(error);
        return res.status(500).json({ message: 'Video upload failed' });
    }
};

module.exports = { uploadVideo };
