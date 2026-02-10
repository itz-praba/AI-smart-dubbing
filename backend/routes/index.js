const express = require('express');

const router = express.Router();

const auth_routes = require('./auth_routes');
const upload_routes = require('./upload_routes');

router.use('',auth_routes);
router.use('',upload_routes);

module.exports = router;