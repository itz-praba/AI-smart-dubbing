# Middleware Documentation

Complete guide to the middleware layer protecting and enhancing your Media Processing API.

## Overview

The API includes **8 middleware components** for production readiness:

1. **Security Headers** - Protects against common web vulnerabilities
2. **GZIP Compression** - Reduces response sizes
3. **Request Logging** - Tracks all API requests
4. **Metrics Collection** - Monitors API performance
5. **Rate Limiting** - Prevents API abuse
6. **Request Size Limits** - Prevents memory exhaustion
7. **Timeout Protection** - Prevents long-running requests
8. **API Key Auth** (Optional) - Restricts API access

## Quick Start

Middleware is automatically enabled when you start the API:

```bash
python main.py
```

All middleware except API Key Authentication is enabled by default.

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Request Settings
REQUEST_TIMEOUT=900
MAX_REQUEST_SIZE_MB=10

# API Key Authentication (Optional)
API_KEY_ENABLED=false
API_KEYS=secret-key-1,secret-key-2

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Programmatic Configuration

Edit `main.py`:

```python
middleware_config = {
    "rate_limit": {
        "enabled": True,
        "requests_per_minute": 120,  # Increase for high-traffic
        "requests_per_hour": 2000
    },
    "timeout": {
        "enabled": True,
        "timeout_seconds": 1800  # 30 minutes
    }
}
setup_middleware(app, middleware_config)
```

## Middleware Details

### 1. Security Headers

**Purpose:** Adds security headers to prevent XSS, clickjacking, and other attacks.

**Headers Added:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`

**Configuration:** Always enabled, no configuration needed.

### 2. GZIP Compression

**Purpose:** Compresses responses to reduce bandwidth.

**Settings:**
- Minimum size: 1000 bytes
- Compression level: Default

**Configuration:** Always enabled.

### 3. Request Logging

**Purpose:** Logs all requests with timing and client information.

**Log Format:**
```
Request started | ID: 1234567890-123456 | Method: POST | Path: /ai/video-to-audio | Client: 192.168.1.1
Request completed | ID: 1234567890-123456 | Status: 200 | Duration: 45.234s
```

**Response Headers Added:**
- `X-Request-ID`: Unique request identifier
- `X-Process-Time`: Request processing time in seconds

**Configuration:** Always enabled.

### 4. Metrics Collection

**Purpose:** Collects API usage statistics.

**Metrics Tracked:**
- Total requests per endpoint
- Error counts per endpoint
- Average response time per endpoint

**Access Metrics:**
```bash
curl http://localhost:8000/metrics
```

**Response Example:**
```json
{
  "total_requests": 1543,
  "total_errors": 12,
  "requests_by_path": {
    "/ai/video-to-audio": 234,
    "/ai/speech-to-text": 456
  },
  "errors_by_path": {
    "/ai/video-to-audio": 3
  },
  "avg_duration_by_path": {
    "/ai/video-to-audio": 45.234,
    "/ai/speech-to-text": 120.456
  }
}
```

**Configuration:** Always enabled.

### 5. Rate Limiting

**Purpose:** Prevents API abuse by limiting requests per IP address.

**Default Limits:**
- 60 requests per minute
- 1000 requests per hour

**Response Headers:**
- `X-RateLimit-Limit-Minute`: Maximum requests per minute
- `X-RateLimit-Limit-Hour`: Maximum requests per hour
- `X-RateLimit-Remaining-Minute`: Remaining requests this minute
- `X-RateLimit-Remaining-Hour`: Remaining requests this hour

**When Limit Exceeded:**
```json
{
  "error": "Rate limit exceeded",
  "detail": "Maximum 60 requests per minute allowed",
  "retry_after": 60
}
```

**Configuration:**
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

**Disable Rate Limiting:**
```bash
RATE_LIMIT_ENABLED=false
```

**Exempt Paths:**
- `/health`
- `/docs`
- `/redoc`
- `/openapi.json`

### 6. Request Size Limit

**Purpose:** Prevents memory exhaustion from large request bodies.

**Default Limit:** 10MB

**When Limit Exceeded:**
```json
{
  "error": "Request body too large",
  "max_size_mb": 10,
  "your_size_mb": 25.5
}
```

**Configuration:**
```bash
MAX_REQUEST_SIZE_MB=10
```

**Increase for Large Files:**
```bash
MAX_REQUEST_SIZE_MB=100  # 100MB
```

**Note:** This limits the request body size, not S3 file sizes (which have separate limits in each controller).

### 7. Timeout Protection

**Purpose:** Prevents requests from running indefinitely.

**Default Timeout:** 900 seconds (15 minutes)

**When Timeout Occurs:**
```json
{
  "error": "Request timeout",
  "detail": "Request exceeded 900 seconds",
  "path": "/ai/video-to-audio"
}
```

**Configuration:**
```bash
REQUEST_TIMEOUT=900
```

**Adjust for Long Operations:**
```bash
REQUEST_TIMEOUT=1800  # 30 minutes
```

### 8. API Key Authentication (Optional)

**Purpose:** Restricts API access to authorized users.

**Default:** Disabled

**Enable API Key Auth:**
```bash
API_KEY_ENABLED=true
API_KEYS=secret-key-1,secret-key-2,secret-key-3
```

**Making Requests:**
```bash
curl -H "X-API-Key: secret-key-1" http://localhost:8000/ai/video-to-audio
```

**Unauthorized Response:**
```json
{
  "error": "Unauthorized",
  "detail": "Valid API key required. Provide X-API-Key header."
}
```

**Exempt Paths:**
- `/health`
- `/docs`
- `/redoc`
- `/openapi.json`

**Generate Secure API Keys:**
```python
import secrets
api_key = secrets.token_urlsafe(32)
print(api_key)
```

## Production Best Practices

### 1. Enable API Key Authentication

```bash
API_KEY_ENABLED=true
API_KEYS=<use-secrets.token_urlsafe(32)>
```

### 2. Restrict CORS Origins

```bash
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

**Never use `*` in production!**

### 3. Adjust Rate Limits

Based on your expected traffic:

**Low traffic (personal/testing):**
```bash
RATE_LIMIT_PER_MINUTE=30
RATE_LIMIT_PER_HOUR=500
```

**Medium traffic (small business):**
```bash
RATE_LIMIT_PER_MINUTE=120
RATE_LIMIT_PER_HOUR=2000
```

**High traffic (enterprise):**
```bash
RATE_LIMIT_PER_MINUTE=300
RATE_LIMIT_PER_HOUR=5000
```

### 4. Configure Timeouts

Based on your operations:

**Mostly short operations:**
```bash
REQUEST_TIMEOUT=600  # 10 minutes
```

**Long video processing:**
```bash
REQUEST_TIMEOUT=1800  # 30 minutes
```

### 5. Monitor Metrics

Set up regular monitoring:

```bash
# Check metrics every 5 minutes
*/5 * * * * curl -s http://localhost:8000/metrics | jq
```

## Advanced Configuration

### Custom Rate Limiting

Create custom rate limiter in `main.py`:

```python
from middleware import RateLimitMiddleware

# Different limits for different endpoints
class CustomRateLimiter(RateLimitMiddleware):
    def __init__(self, app):
        super().__init__(app, requests_per_minute=60, requests_per_hour=1000)
    
    async def dispatch(self, request, call_next):
        # Higher limits for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Lower limits for expensive operations
        if request.url.path == "/ai/video-to-audio":
            self.requests_per_minute = 10
            self.requests_per_hour = 100
        
        return await super().dispatch(request, call_next)

app.add_middleware(CustomRateLimiter)
```

### IP Whitelisting

Add to `middleware.py`:

```python
WHITELISTED_IPS = {"192.168.1.1", "10.0.0.1"}

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        
        if client_ip not in WHITELISTED_IPS:
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        return await call_next(request)
```

### Request ID Tracking

Request IDs are automatically added to responses:

```bash
curl -v http://localhost:8000/health
# Response headers:
# X-Request-ID: 1234567890-123456
# X-Process-Time: 0.003
```

Use request IDs for debugging:

```python
import logging
logger.info(f"Processing job | Request-ID: {request_id}")
```

## Monitoring and Alerting

### Log Analysis

All requests are logged. Parse logs for analysis:

```bash
# Count requests by path
grep "Request completed" app.log | awk '{print $8}' | sort | uniq -c

# Find slow requests (>10s)
grep "Request completed" app.log | awk '$12 > 10.0'

# Count errors
grep "Request failed" app.log | wc -l
```

### Prometheus Integration (Optional)

For production, consider Prometheus client:

```bash
pip install prometheus-client
```

```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.get("/prometheus-metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Troubleshooting

### High Rate of 429 Errors

**Problem:** Users hitting rate limits frequently.

**Solution:**
1. Increase rate limits in `.env`
2. Implement caching on client side
3. Use batch endpoints where available

### High Memory Usage

**Problem:** Middleware consuming too much memory.

**Solution:**
1. Reduce metrics history (modify MetricsMiddleware)
2. Implement log rotation
3. Use external metrics collector (Prometheus)

### Slow Response Times

**Problem:** Middleware adding latency.

**Solution:**
1. Disable non-essential middleware
2. Use async/await properly
3. Profile middleware execution time

### Rate Limiting Not Working

**Problem:** Users bypassing rate limits.

**Solution:**
1. Check proxy configuration (X-Forwarded-For)
2. Verify IP extraction logic
3. Consider using Redis for distributed rate limiting

## Testing Middleware

### Test Rate Limiting

```bash
# Send 100 requests rapidly
for i in {1..100}; do
  curl http://localhost:8000/health &
done
wait

# Should see 429 errors after limit
```

### Test Request Size Limit

```bash
# Generate large request
dd if=/dev/zero of=large.bin bs=1M count=20

# Should fail with 413 error
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  --data-binary @large.bin \
  http://localhost:8000/ai/video-to-audio
```

### Test API Key Auth

```bash
# Without API key (should fail)
curl http://localhost:8000/ai/video-to-audio

# With API key (should succeed)
curl -H "X-API-Key: your-key" http://localhost:8000/ai/video-to-audio
```

### Test Timeout

```bash
# Create endpoint that sleeps
# Should timeout after configured duration
```

## Summary

Your API now has **enterprise-grade middleware** providing:

✅ **Security** - Headers, size limits, optional authentication  
✅ **Performance** - GZIP compression, timeout protection  
✅ **Monitoring** - Request logging, metrics collection  
✅ **Protection** - Rate limiting, request validation  

All middleware is production-ready and configurable via environment variables.