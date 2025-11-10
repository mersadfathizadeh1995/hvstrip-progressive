# HVSR MCP Server - Complete Resources & References Guide

## Table of Contents

1. [MCP Protocol Specifications](#mcp-protocol-specifications)
2. [Python SDK & Frameworks](#python-sdk-frameworks)
3. [Claude Desktop Integration](#claude-desktop-integration)
4. [HVSR Scientific Background](#hvsr-scientific-background)
5. [Development Tools & Testing](#development-tools-testing)
6. [Deployment & Production](#deployment-production)
7. [Community Resources](#community-resources)

---

## MCP Protocol Specifications

### Official Documentation

**Primary Resource**: [Model Context Protocol Official Site](https://modelcontextprotocol.io)
- Complete protocol specification
- Architecture overview
- Best practices guide

**Specification Details**: [MCP Specification](https://modelcontextprotocol.io/specification)
- JSON-RPC 2.0 foundation
- Message format definitions
- Transport mechanisms (stdio, HTTP/SSE, WebSocket)
- Schema definitions

**GitHub Organization**: [Model Context Protocol](https://github.com/modelcontextprotocol)
- Official repositories
- Example servers
- Community contributions

### Key Concepts

#### Architecture Components

**Host**: Central coordinator (e.g., Claude Desktop)
- Manages multiple clients
- Handles security and permissions
- Coordinates LLM interactions

**Client**: Protocol intermediary
- 1:1 relationship with server
- Session management
- Message routing

**Server**: Resource provider
- Exposes tools (functions)
- Provides resources (data)
- Defines prompts (templates)

#### Core Features

**Tools**: Actions the AI can execute
- JSON Schema parameter definitions
- Synchronous or asynchronous execution
- Type-safe invocations
- Return structured results

**Resources**: Data access endpoints
- URI-based addressing (e.g., `resource://type/{id}`)
- Static or dynamic content
- MIME type support
- Template parameters with `{placeholders}`

**Prompts**: Reusable instruction templates
- Structured message templates
- Parameter substitution
- Workflow guidance
- Context-aware construction

**Sampling**: Server-initiated LLM calls
- Agentic behaviors
- Recursive processing
- Multi-step reasoning

**Roots**: Filesystem boundary queries
- URI or path-based scoping
- Security constraints
- Access control

**Elicitation**: Server-initiated information requests
- Interactive workflows
- User input collection
- Dynamic parameter gathering

### Protocol Details

**Message Format**: JSON-RPC 2.0
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {"param": "value"}
  }
}
```

**Response Format**:
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "result": {
    "content": [
      {"type": "text", "text": "Result data"}
    ]
  }
}
```

**Transport Options**:
1. **stdio**: Standard input/output (most common for local servers)
2. **HTTP with SSE**: Server-sent events for streaming
3. **WebSocket**: Bidirectional communication

---

## Python SDK & Frameworks

### Official Python SDK

**Repository**: [python-sdk](https://github.com/modelcontextprotocol/python-sdk)
**Installation**: `pip install mcp`

**Basic Usage**:
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Tool description",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        result = process(arguments["param"])
        return [TextContent(type="text", text=result)]
```

### FastMCP Framework

**Repository**: [FastMCP](https://github.com/jlowin/fastmcp)
**PyPI**: [fastmcp](https://pypi.org/project/fastmcp/)
**Installation**: `pip install fastmcp`

**Key Features**:
- Decorator-based API (similar to Flask/FastAPI)
- Automatic schema generation from type hints
- Pydantic model support
- Async/await support
- Built-in error handling

**Quick Start**:
```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("config://version")
def get_version() -> str:
    return "1.0.0"

if __name__ == "__main__":
    mcp.run()  # Defaults to stdio transport
```

**Advanced Features**:
```python
from pydantic import BaseModel

class UserInput(BaseModel):
    name: str
    age: int

@mcp.tool()
async def create_user(user: UserInput) -> dict:
    """Create a user with Pydantic validation"""
    # Automatic validation of user.name and user.age
    return {"id": 123, "name": user.name}

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Dynamic resource with template parameter"""
    return {"user_id": user_id, "data": "..."}
```

### Environment Setup

**Using UV (Recommended)**:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install fastmcp
```

**Using pip**:
```bash
python -m venv venv
source venv/bin/activate
pip install fastmcp
```

---

## Claude Desktop Integration

### Configuration Files

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

### Access Methods

1. **Claude Desktop UI**:
   - Open Claude Desktop
   - Go to Settings > Developer
   - Click "Edit Config"
   - Modify JSON configuration
   - Restart Claude Desktop

2. **Direct File Edit**:
   - Navigate to config file location
   - Edit with text editor
   - Save changes
   - Restart Claude Desktop

### Configuration Examples

**Basic stdio Configuration**:
```json
{
  "mcpServers": {
    "hvsr-analysis": {
      "command": "python",
      "args": ["-m", "hvsr_mcp_server"],
      "env": {
        "HVF_EXE_PATH": "/path/to/HVf.exe"
      }
    }
  }
}
```

**UV-based Configuration**:
```json
{
  "mcpServers": {
    "hvsr-analysis": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/project",
        "run",
        "python",
        "-m",
        "hvsr_mcp_server"
      ],
      "env": {
        "HVF_EXE_PATH": "/path/to/HVf.exe",
        "HVSR_WORKSPACE": "/path/to/workspace"
      }
    }
  }
}
```

**HTTP/SSE Configuration**:
```json
{
  "mcpServers": {
    "hvsr-analysis": {
      "url": "https://your-server.com/mcp",
      "apiKey": "your-api-key"
    }
  }
}
```

### Troubleshooting

**Common Issues**:

1. **Server not appearing in Claude**
   - Check config file syntax (valid JSON)
   - Verify command path is absolute
   - Restart Claude Desktop completely
   - Check logs at: `~/Library/Logs/Claude/mcp*.log` (macOS)

2. **"uv not found" error**
   - Use absolute path: `which uv` (Unix) or `where uv` (Windows)
   - Update config with full path

3. **Environment variables not working**
   - Use absolute paths in env values
   - Check PATH includes required executables

**Verification**:
```bash
# Test server manually
python -m hvsr_mcp_server

# Or with UV
uv run python -m hvsr_mcp_server
```

---

## HVSR Scientific Background

### Method Overview

**HVSR (Horizontal-to-Vertical Spectral Ratio)**: Passive seismic method for site characterization
- Uses ambient seismic noise (microtremors)
- Estimates fundamental resonance frequency
- Determines sediment thickness over bedrock
- Calculates average shear-wave velocity

### Key References

**Foundational Papers**:
1. Nakamura, Y. (1989). "A method for dynamic characteristics estimation of subsurface using microtremor on the ground surface."
2. Field, E.H., & Jacob, K.H. (1993). "The theoretical response of sedimentary layers to ambient seismic noise."

**Recent Applications**:
- Multi-method site characterization (Palmer & Atkinson, 2020)
- Glacier thickness estimation (Picotti et al., 2017)
- Seismic microzonation studies
- Infrastructure characterization

### Progressive Layer Stripping Method

**Concept**: Systematically remove layers from velocity model to identify which interfaces control HVSR peaks

**Workflow**:
1. Start with complete velocity model
2. Compute HVSR curve
3. Remove deepest finite layer
4. Recompute HVSR curve
5. Analyze peak evolution
6. Identify controlling interfaces

**Key Parameters**:
- **Vs**: Shear-wave velocity (m/s)
- **Vp**: Compressional-wave velocity (m/s)
- **ρ**: Density (kg/m³)
- **h**: Layer thickness (m)
- **f0**: Fundamental resonance frequency (Hz)

**Relationship**: `f0 = Vs / (4h)` for simple case

### HV-INV Solver

**Repository**: [HV-INV](https://github.com/agarcia-jerez/HV-INV)
**License**: GPL-3.0

**Capabilities**:
- Forward HVSR modeling using diffuse field theory
- Inverse problem solving
- Multiple layer support
- Frequency-dependent calculations

**Citation**: García-Jerez, A., et al. (2016). "A computer code for forward calculation and inversion of the H/V spectral ratio under the diffuse field assumption."

### Site Characterization Applications

**Use Cases**:
1. **Earthquake engineering**: Site amplification studies
2. **Geotechnical projects**: Foundation design
3. **Infrastructure assessment**: Bridge, dam integrity
4. **Geological mapping**: Bedrock depth mapping
5. **Resource exploration**: Sediment characterization

---

## Development Tools & Testing

### MCP Inspector

**Purpose**: Visual testing tool for MCP servers
**Repository**: [mcp-inspector](https://github.com/modelcontextprotocol/inspector)

**Usage**:
```bash
npx @modelcontextprotocol/inspector python -m hvsr_mcp_server
```

**Features**:
- Interactive tool testing
- Resource exploration
- Prompt preview
- Real-time debugging

### Testing Frameworks

**pytest-asyncio**:
```bash
pip install pytest-asyncio
```

**Example Test**:
```python
import pytest
from mcp import Client

@pytest.mark.asyncio
async def test_hvsr_analysis():
    async with Client("python -m hvsr_mcp_server") as client:
        # Test tool call
        result = await client.call_tool(
            "run_progressive_analysis",
            {
                "model_path": "test_model.txt",
                "output_dir": "test_output",
                "hvf_exe_path": "/path/to/HVf"
            }
        )
        assert result["status"] == "success"
```

### Logging and Debugging

**Python Logging**:
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hvsr_mcp.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("hvsr-mcp")
```

**MCP Logging**:
```python
from mcp.types import LoggingLevel

# Send logs to client
await server.send_log_message(
    level=LoggingLevel.INFO,
    message="Analysis started",
    logger="hvsr-mcp"
)
```

---

## Deployment & Production

### Docker Containerization

**Dockerfile Example**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY hvsr_mcp_server/ ./hvsr_mcp_server/

# Expose port for HTTP/SSE transport
EXPOSE 8080

CMD ["python", "-m", "hvsr_mcp_server"]
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  hvsr-mcp:
    build: .
    environment:
      - HVF_EXE_PATH=/usr/local/bin/HVf
    volumes:
      - ./workspace:/workspace
    ports:
      - "8080:8080"
```

### Production Best Practices

**Configuration Management**:
```python
from pydantic_settings import BaseSettings

class MCPConfig(BaseSettings):
    hvf_exe_path: str
    workspace_dir: str = "./workspace"
    max_concurrent_analyses: int = 5
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "HVSR_MCP_"
        env_file = ".env"
```

**Resource Limits**:
```python
# Implement timeouts
import asyncio

@mcp.tool()
async def run_analysis(...):
    try:
        async with asyncio.timeout(300):  # 5 minute timeout
            result = await perform_analysis(...)
            return result
    except asyncio.TimeoutError:
        return {"error": "Analysis timeout exceeded"}
```

**Rate Limiting**:
```python
from collections import defaultdict
import time

request_counts = defaultdict(list)

async def rate_limit(user_id: str, max_requests: int = 10, window: int = 60):
    now = time.time()
    requests = request_counts[user_id]
    
    # Remove old requests
    requests[:] = [t for t in requests if now - t < window]
    
    if len(requests) >= max_requests:
        raise Exception("Rate limit exceeded")
    
    requests.append(now)
```

### Monitoring

**Health Checks**:
```python
@mcp.tool()
def health_check() -> dict:
    """Server health status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "hvf_available": check_hvf_available(),
        "disk_space": get_disk_space(),
        "active_analyses": get_active_count()
    }
```

**Metrics Collection**:
```python
from prometheus_client import Counter, Histogram

analysis_counter = Counter(
    'hvsr_analyses_total',
    'Total number of analyses run'
)

analysis_duration = Histogram(
    'hvsr_analysis_duration_seconds',
    'Time spent running analyses'
)

@analysis_duration.time()
async def run_analysis(...):
    analysis_counter.inc()
    # ... perform analysis
```

---

## Community Resources

### Official MCP Resources

**Documentation**: https://modelcontextprotocol.io/docs
**Discussions**: https://github.com/modelcontextprotocol/specification/discussions
**Discord**: Check MCP GitHub for community links

### Example Servers

**Official Examples**: [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- Filesystem server
- GitHub integration
- Database connectors
- Web scraping tools

**Community Servers**: [MCP Registry](https://github.com/modelcontextprotocol/registry)
- Browse available servers
- Submit your own
- Learn from others' implementations

### Learning Resources

**Tutorials**:
1. [Building MCP Servers with Python](https://scrapfly.io/blog/mcp-server-python/)
2. [FastMCP Complete Guide](https://mcpcat.io/guides/build-python-fastmcp)
3. [MCP Architecture Deep Dive](https://getknit.dev/blog/mcp-architecture)

**Video Resources**:
- YouTube: "Claude MCP Server Tutorial"
- YouTube: "Building AI Tools with MCP"

**Blog Posts**:
- [MCP Best Practices](https://modelcontextprotocol.info/best-practices)
- [Understanding MCP Features](https://workos.com/blog/mcp-features)
- [MCP Deployment Guide](https://northflank.com/blog/mcp-deployment)

### Related Projects

**HV-INV Family**:
- [HV-INV](https://github.com/agarcia-jerez/HV-INV): MATLAB-based HVSR tool
- [HV-DFA](https://github.com/agarcia-jerez/HV-DFA): Diffuse field approximation

**Geophysical Tools**:
- SeismoVLAB: Seismic analysis
- OpenSees: Structural analysis
- MASW tools: Surface wave analysis

### Getting Help

**Stack Overflow**: Tag questions with `model-context-protocol`
**GitHub Issues**: Open issues in relevant repositories
**Email Support**: Contact package maintainers directly

---

## Quick Reference Cheat Sheet

### MCP Server Basics

```python
from fastmcp import FastMCP

mcp = FastMCP("server-name")

@mcp.tool()  # Define tool
@mcp.resource("uri://pattern")  # Define resource
@mcp.prompt()  # Define prompt

mcp.run()  # Start server (stdio by default)
```

### Claude Desktop Config

```json
{
  "mcpServers": {
    "name": {
      "command": "python",
      "args": ["-m", "module"],
      "env": {"VAR": "value"}
    }
  }
}
```

### Common Commands

```bash
# Install FastMCP
pip install fastmcp

# Run server
fastmcp run server.py

# Test with inspector
npx @modelcontextprotocol/inspector python server.py

# Check Claude config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

---

## Conclusion

This resource guide provides comprehensive information for developing your HVSR MCP server. Refer to specific sections as needed during development, and don't hesitate to explore the linked resources for deeper understanding.

For questions or contributions, contact the project maintainers or open issues on GitHub.

**Happy coding!** 🎉
