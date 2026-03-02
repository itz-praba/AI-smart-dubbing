"""
Middleware for Media Processing API
Includes: Security, Rate Limiting, Request Logging, Error Handling, Monitoring
"""
import time
import logging
import traceback
from typing import Callable
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

# =====================================================
# RATE LIMITING MIDDLEWARE
# =====================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse
    Limits requests per IP address
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = defaultdict(list)
        self.hour_requests = defaultdict(list)
        self.lock = Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        
        current_time = datetime.now()
        
        with self.lock:
            # Clean old requests
            minute_ago = current_time - timedelta(minutes=1)
            hour_ago = current_time - timedelta(hours=1)
            
            self.minute_requests[client_ip] = [
                t for t in self.minute_requests[client_ip] if t > minute_ago
            ]
            self.hour_requests[client_ip] = [
                t for t in self.hour_requests[client_ip] if t > hour_ago
            ]
            
            # Check limits
            if len(self.minute_requests[client_ip]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded (minute) for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Maximum {self.requests_per_minute} requests per minute allowed",
                        "retry_after": 60
                    }
                )
            
            if len(self.hour_requests[client_ip]) >= self.requests_per_hour:
                logger.warning(f"Rate limit exceeded (hour) for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Maximum {self.requests_per_hour} requests per hour allowed",
                        "retry_after": 3600
                    }
                )
            
            # Add current request
            self.minute_requests[client_ip].append(current_time)
            self.hour_requests[client_ip].append(current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            self.requests_per_minute - len(self.minute_requests[client_ip])
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            self.requests_per_hour - len(self.hour_requests[client_ip])
        )
        
        return response


# =====================================================
# REQUEST LOGGING MIDDLEWARE
# =====================================================
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all incoming requests with timing information
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = f"{int(time.time() * 1000)}-{id(request)}"
        
        # Start timer
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        
        # Log request
        logger.info(
            f"Request started | ID: {request_id} | "
            f"Method: {request.method} | Path: {request.url.path} | "
            f"Client: {client_ip}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed | ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.3f}s"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{duration:.3f}"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Request failed | ID: {request_id} | "
                f"Error: {str(e)} | Duration: {duration:.3f}s",
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "detail": str(e)
                }
            )


# =====================================================
# SECURITY HEADERS MIDDLEWARE
# =====================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


# =====================================================
# REQUEST SIZE LIMIT MIDDLEWARE
# =====================================================
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Limits the size of incoming requests to prevent memory exhaustion
    """
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            
            if content_length > self.max_request_size:
                logger.warning(
                    f"Request body too large: {content_length} bytes "
                    f"(max: {self.max_request_size} bytes)"
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "Request body too large",
                        "max_size_mb": self.max_request_size / (1024 * 1024),
                        "your_size_mb": content_length / (1024 * 1024)
                    }
                )
        
        return await call_next(request)


# =====================================================
# TIMEOUT MIDDLEWARE
# =====================================================
class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Adds timeout to prevent long-running requests from blocking workers
    Note: This is a soft timeout - actual processing may continue
    """
    
    def __init__(self, app, timeout_seconds: int = 900):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Direct execution without timeout
        return await call_next(request)

# =====================================================
# METRICS MIDDLEWARE
# =====================================================
class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Collects basic metrics about API usage
    In production, consider using Prometheus client instead
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.total_duration = defaultdict(float)
        self.lock = Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        path = request.url.path
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            
            with self.lock:
                self.request_count[path] += 1
                self.total_duration[path] += duration
                
                if response.status_code >= 400:
                    self.error_count[path] += 1
            
            return response
            
        except Exception as e:
            with self.lock:
                self.error_count[path] += 1
            raise
    
    def get_metrics(self):
        """Get current metrics (useful for a /metrics endpoint)"""
        with self.lock:
            return {
                "total_requests": sum(self.request_count.values()),
                "total_errors": sum(self.error_count.values()),
                "requests_by_path": dict(self.request_count),
                "errors_by_path": dict(self.error_count),
                "avg_duration_by_path": {
                    path: self.total_duration[path] / self.request_count[path]
                    for path in self.request_count
                    if self.request_count[path] > 0
                }
            }


# =====================================================
# API KEY AUTHENTICATION MIDDLEWARE (Optional)
# =====================================================
class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Simple API key authentication
    Enable this in production for additional security
    """
    
    def __init__(self, app, api_keys: list = None, enabled: bool = False):
        super().__init__(app)
        self.api_keys = set(api_keys or [])
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Skip authentication for public endpoints
        public_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("X-API-Key")
        
        if not api_key or api_key not in self.api_keys:
            logger.warning(f"Unauthorized access attempt from {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "detail": "Valid API key required. Provide X-API-Key header."
                }
            )
        
        return await call_next(request)


# =====================================================
# CORS MIDDLEWARE (Enhanced)
# =====================================================
def get_cors_middleware():
    """
    Returns configured CORS middleware
    Adjust allowed_origins for production
    """
    return CORSMiddleware


# =====================================================
# HELPER: Add All Middleware to App
# =====================================================
def setup_middleware(app, config: dict = None):
    """
    Setup all middleware for the application
    
    Args:
        app: FastAPI application instance
        config: Configuration dictionary with middleware settings
    
    Example config:
    {
        "rate_limit": {"enabled": True, "requests_per_minute": 60},
        "api_key": {"enabled": False, "keys": ["secret-key-1"]},
        "timeout": {"enabled": True, "timeout_seconds": 900},
        "request_size_limit": {"enabled": True, "max_size_mb": 10}
    }
    """
    config = config or {}
    
    # 1. Security Headers (always enabled)
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("✓ Security headers middleware enabled")
    
    # 2. GZIP Compression (always enabled)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    logger.info("✓ GZIP compression middleware enabled")
    
    # 3. Request Logging (always enabled)
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("✓ Request logging middleware enabled")
    
    # 4. Metrics Collection (always enabled)
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware.dispatch)
    logger.info("✓ Metrics middleware enabled")
    
    # Store metrics instance for later access
    app.state.metrics = metrics_middleware
    
    # 5. Rate Limiting (configurable)
    rate_limit_config = config.get("rate_limit", {"enabled": True})
    if rate_limit_config.get("enabled", True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rate_limit_config.get("requests_per_minute", 60),
            requests_per_hour=rate_limit_config.get("requests_per_hour", 1000)
        )
        logger.info(
            f"✓ Rate limiting enabled: "
            f"{rate_limit_config.get('requests_per_minute', 60)}/min, "
            f"{rate_limit_config.get('requests_per_hour', 1000)}/hour"
        )
    
    # 6. Request Size Limit (configurable)
    size_limit_config = config.get("request_size_limit", {"enabled": True})
    if size_limit_config.get("enabled", True):
        max_size_mb = size_limit_config.get("max_size_mb", 10)
        app.add_middleware(
            RequestSizeLimitMiddleware,
            max_request_size=max_size_mb * 1024 * 1024
        )
        logger.info(f"✓ Request size limit enabled: {max_size_mb}MB")
    
    # 7. Timeout DISABLED
    logger.info("✗ Timeout middleware disabled")
    
    # 8. API Key Authentication (optional, disabled by default)
    api_key_config = config.get("api_key", {"enabled": False})
    if api_key_config.get("enabled", False):
        app.add_middleware(
            APIKeyMiddleware,
            api_keys=api_key_config.get("keys", []),
            enabled=True
        )
        logger.info("✓ API key authentication enabled")
    else:
        logger.info("✗ API key authentication disabled (enable for production)")
    
    logger.info("=" * 60)
    logger.info("All middleware configured successfully")
    logger.info("=" * 60)