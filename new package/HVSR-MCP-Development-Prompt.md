# HVSR Progressive Analysis MCP Server - Development Prompt

## Project Overview

You are tasked with creating a **Model Context Protocol (MCP) server** for the `hvstrip-progressive` package, which performs progressive layer stripping analysis of Horizontal-to-Vertical Spectral Ratio (HVSR) data for geotechnical site characterization.

### Primary Objective

Transform the existing `hvstrip-progressive` Python package into a fully-functional MCP server that can be integrated with Claude Desktop and other MCP clients, enabling AI-assisted HVSR analysis and research workflows.

---

## Package Context

### Current Package Capabilities

The `hvstrip-progressive` package (located at `https://github.com/mersadfathizadeh1995/hvstrip-progressive`) provides:

1. **Progressive Layer Stripping**: Systematic removal of deepest finite layers from velocity models
2. **HV Forward Modeling**: Integration with external HVf solver (from HV-INV project)
3. **Post-Processing**: Publication-ready plots and analysis
4. **Batch Workflow**: Complete end-to-end analysis pipeline
5. **Report Generation**: Comprehensive multi-panel figures, CSV summaries, and metadata

### Core Modules to Wrap

- `stripper.py` - Layer removal logic
- `hv_forward.py` - HVf executable interface
- `hv_postprocess.py` - Visualization and analysis
- `batch_workflow.py` - Workflow orchestration
- `report_generator.py` - Report creation

### External Dependencies

- **HVf executable**: External solver from HV-INV project (GPL-3.0)
- **Python libraries**: numpy, scipy, matplotlib, pandas

---

## MCP Architecture Requirements

### Protocol Specifications

**Foundation**: JSON-RPC 2.0 message format
**Transport**: Support stdio (primary) and optionally HTTP/SSE
**SDK**: Use official Python MCP SDK or FastMCP framework

### Three-Tier Architecture

1. **Host**: Claude Desktop (or other MCP clients)
2. **Client**: MCP protocol handler (managed by host)
3. **Server**: Your HVSR analysis MCP server

---

## Implementation Requirements

### 1. Server Structure

```python
from mcp.server.fastmcp import FastMCP
import logging

# Initialize MCP server
mcp = FastMCP(
    name="HVSR-Progressive-Analysis",
    version="1.0.0",
    dependencies=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "hvstrip-progressive"
    ]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hvsr-mcp-server")
```

### 2. Tools to Implement

#### Core Analysis Tools

**Tool 1: run_progressive_analysis**
```python
@mcp.tool()
async def run_progressive_analysis(
    model_path: str,
    output_dir: str,
    hvf_exe_path: str,
    freq_min: float = 0.5,
    freq_max: float = 20.0
) -> dict:
    """
    Execute complete progressive layer stripping workflow.
    
    Args:
        model_path: Path to input velocity model file
        output_dir: Directory for output files
        hvf_exe_path: Path to HVf executable
        freq_min: Minimum frequency for HVSR analysis (Hz)
        freq_max: Maximum frequency for HVSR analysis (Hz)
    
    Returns:
        Dictionary containing analysis summary and output file paths
    """
    # Implementation using batch_workflow.run_complete_workflow()
    pass
```

**Tool 2: compute_hv_curve**
```python
@mcp.tool()
def compute_hv_curve(
    model_path: str,
    hvf_exe_path: str,
    output_csv: str
) -> dict:
    """
    Calculate HVSR curve for a given velocity model.
    
    Args:
        model_path: Path to velocity model file
        hvf_exe_path: Path to HVf executable
        output_csv: Path for output CSV file
    
    Returns:
        Dictionary with peak frequencies and curve data
    """
    # Implementation using hv_forward module
    pass
```

**Tool 3: analyze_peak_evolution**
```python
@mcp.tool()
def analyze_peak_evolution(
    strip_directory: str
) -> dict:
    """
    Analyze how HVSR peaks change through layer stripping.
    
    Args:
        strip_directory: Directory containing strip results
    
    Returns:
        Peak evolution analysis and controlling interfaces
    """
    # Implementation using hv_postprocess module
    pass
```

**Tool 4: generate_comprehensive_report**
```python
@mcp.tool()
def generate_comprehensive_report(
    strip_directory: str,
    output_dir: str,
    include_pdf: bool = True
) -> dict:
    """
    Generate publication-quality analysis report.
    
    Args:
        strip_directory: Directory with strip results
        output_dir: Output directory for reports
        include_pdf: Whether to generate PDF version
    
    Returns:
        Paths to generated report files
    """
    # Implementation using report_generator module
    pass
```

**Tool 5: validate_velocity_model**
```python
@mcp.tool()
def validate_velocity_model(
    model_path: str
) -> dict:
    """
    Validate velocity model format and parameters.
    
    Args:
        model_path: Path to velocity model file
    
    Returns:
        Validation results and model properties
    """
    # Check model format, layer properties, etc.
    pass
```

### 3. Resources to Expose

```python
@mcp.resource("hvsr://models/{model_id}")
def get_velocity_model(model_id: str) -> str:
    """
    Retrieve stored velocity model by ID.
    
    Returns model file content in HVf format.
    """
    pass

@mcp.resource("hvsr://results/{analysis_id}/summary")
def get_analysis_summary(analysis_id: str) -> str:
    """
    Get JSON summary of completed analysis.
    
    Includes peak frequencies, layer properties, and key findings.
    """
    pass

@mcp.resource("hvsr://results/{analysis_id}/plots/{plot_type}")
def get_analysis_plot(analysis_id: str, plot_type: str) -> str:
    """
    Retrieve specific plot from analysis results.
    
    plot_type: hv_curves, peak_evolution, waterfall, etc.
    Returns base64-encoded image or file path.
    """
    pass

@mcp.resource("hvsr://examples/soil_profile")
def get_example_model() -> str:
    """
    Get example soil profile model for testing.
    
    Returns the example model from examples/soil_profile/model.txt
    """
    pass
```

### 4. Prompts to Define

```python
@mcp.prompt()
def analyze_site_prompt(
    location: str,
    geology_description: str,
    expected_depth_to_bedrock: str
) -> list:
    """
    Guided workflow for site characterization.
    
    Provides step-by-step instructions for:
    1. Model preparation
    2. Analysis execution
    3. Result interpretation
    4. Report generation
    """
    return [
        {
            "role": "user",
            "content": f"""I need to characterize a site at {location}.
            
            Geology: {geology_description}
            Expected bedrock depth: {expected_depth_to_bedrock}
            
            Please guide me through HVSR progressive analysis."""
        },
        {
            "role": "assistant",
            "content": """I'll help you perform HVSR progressive layer stripping analysis.
            
            Step 1: Prepare your velocity model
            - Create a text file with layer parameters (thickness, Vs, Vp, density)
            - Ensure proper format for HVf solver
            
            Step 2: Run the analysis
            Use the run_progressive_analysis tool with your model file.
            
            Step 3: Interpret results
            We'll analyze peak evolution and identify controlling interfaces.
            
            Step 4: Generate report
            Create publication-ready figures and summary.
            
            Let's start. Do you have a velocity model ready?"""
        }
    ]

@mcp.prompt()
def interpret_peaks_prompt(
    peak_frequencies: str,
    model_layers: str
) -> list:
    """
    Help interpret HVSR peak significance.
    
    Provides guidance on:
    - Peak-interface relationships
    - Impedance contrast analysis
    - Fundamental vs higher modes
    """
    pass

@mcp.prompt()
def troubleshoot_model_prompt(
    error_message: str,
    model_description: str
) -> list:
    """
    Debug velocity model issues.
    
    Common problems:
    - Format errors
    - Unrealistic parameters
    - Solver failures
    """
    pass
```

### 5. Configuration and Deployment

#### Server Configuration

```python
# config/mcp_config.yaml
server:
  name: "hvsr-progressive-mcp"
  version: "1.0.0"
  description: "MCP server for HVSR progressive layer stripping analysis"
  
transport:
  type: "stdio"  # Primary transport
  fallback: "sse"  # Optional HTTP/SSE
  
hvf_solver:
  default_path: null  # Must be configured by user
  required: true
  
directories:
  workspace: "./hvsr_workspace"
  models: "./hvsr_workspace/models"
  results: "./hvsr_workspace/results"
  temp: "./hvsr_workspace/temp"
  
logging:
  level: "INFO"
  format: "json"
  file: "./hvsr_mcp.log"
```

#### Claude Desktop Integration

Create configuration instructions for users:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hvsr-progressive": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/hvsr-mcp-env",
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

### 6. Error Handling and Validation

```python
class HVSRMCPError(Exception):
    """Base exception for HVSR MCP server"""
    pass

class ModelValidationError(HVSRMCPError):
    """Raised when velocity model is invalid"""
    pass

class SolverExecutionError(HVSRMCPError):
    """Raised when HVf solver fails"""
    pass

# Implement comprehensive error handling in all tools
@mcp.tool()
def run_progressive_analysis(...):
    try:
        # Validate inputs
        if not os.path.exists(model_path):
            raise ModelValidationError(f"Model file not found: {model_path}")
        
        if not os.path.exists(hvf_exe_path):
            raise SolverExecutionError(f"HVf executable not found: {hvf_exe_path}")
        
        # Execute workflow
        results = batch_workflow.run_complete_workflow(...)
        
        return {
            "status": "success",
            "results": results
        }
        
    except ModelValidationError as e:
        logger.error(f"Model validation failed: {str(e)}")
        return {
            "status": "error",
            "error_type": "validation",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "error_type": "internal",
            "message": str(e)
        }
```

### 7. Testing and Validation

```python
# tests/test_mcp_server.py
import pytest
from mcp import Client

@pytest.mark.asyncio
async def test_server_connection():
    """Test basic MCP server connectivity"""
    async with Client("path/to/hvsr_mcp_server.py") as client:
        # Test ping
        await client.ping()
        
        # List available tools
        tools = await client.list_tools()
        assert len(tools) > 0
        
        # List resources
        resources = await client.list_resources()
        assert len(resources) > 0

@pytest.mark.asyncio
async def test_progressive_analysis():
    """Test complete analysis workflow"""
    async with Client("path/to/hvsr_mcp_server.py") as client:
        result = await client.call_tool(
            "run_progressive_analysis",
            {
                "model_path": "examples/soil_profile/model.txt",
                "output_dir": "test_output",
                "hvf_exe_path": "/path/to/HVf"
            }
        )
        assert result["status"] == "success"
```

### 8. Documentation Requirements

Create the following documentation files:

1. **README.md**: Overview and quick start
2. **INSTALLATION.md**: Detailed setup instructions
3. **API_REFERENCE.md**: Complete tool/resource/prompt documentation
4. **CLAUDE_INTEGRATION.md**: Claude Desktop setup guide
5. **EXAMPLES.md**: Usage examples and tutorials
6. **TROUBLESHOOTING.md**: Common issues and solutions

---

## Development Workflow

### Phase 1: Core Server Setup (Week 1)
1. Initialize FastMCP server structure
2. Set up configuration management
3. Implement basic tool scaffolding
4. Test local server execution

### Phase 2: Tool Implementation (Week 2-3)
1. Implement all 5 core analysis tools
2. Add comprehensive error handling
3. Create unit tests for each tool
4. Validate with example datasets

### Phase 3: Resources and Prompts (Week 4)
1. Implement resource endpoints
2. Define prompt templates
3. Test resource access patterns
4. Validate prompt workflows

### Phase 4: Integration and Testing (Week 5)
1. Test Claude Desktop integration
2. Create integration test suite
3. Performance optimization
4. Documentation completion

### Phase 5: Deployment and Release (Week 6)
1. Package for distribution (PyPI)
2. Docker containerization
3. Create deployment guides
4. Release v1.0.0

---

## Key Design Principles

### 1. Modularity
- Keep MCP server logic separate from core analysis code
- Use composition over inheritance
- Enable easy extension for future features

### 2. Robustness
- Comprehensive error handling
- Input validation for all tools
- Graceful degradation when HVf unavailable

### 3. User Experience
- Clear, descriptive tool documentation
- Helpful error messages
- Intuitive prompt workflows

### 4. Performance
- Async operations where possible
- Efficient file I/O
- Resource cleanup after operations

### 5. Security
- Path validation to prevent directory traversal
- Safe subprocess execution
- Environment variable management

---

## Success Criteria

The MCP server is considered successful when:

1. ✅ All tools execute correctly with valid inputs
2. ✅ Resources are accessible via URI patterns
3. ✅ Prompts guide users through workflows
4. ✅ Claude Desktop integration works smoothly
5. ✅ Error messages are clear and actionable
6. ✅ Documentation is comprehensive
7. ✅ Tests achieve >80% coverage
8. ✅ Example workflows run end-to-end

---

## Additional Features (Future Enhancements)

### Research Ideas from "new package" folder

Based on the repository structure, consider implementing:

1. **Batch Processing**: Multiple site analyses in parallel
2. **Model Comparison**: Side-by-side comparison tools
3. **Parameter Sensitivity**: Automated sensitivity analysis
4. **ML Integration**: Machine learning for peak identification
5. **Web Interface**: Optional web-based visualization
6. **Database Backend**: Store analysis history
7. **Collaborative Features**: Share models and results
8. **Real-time Monitoring**: Progress tracking for long analyses

---

## References and Resources

### Official Documentation
- **MCP Specification**: https://modelcontextprotocol.io/specification
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **FastMCP Documentation**: https://github.com/jlowin/fastmcp

### Example Servers
- **MCP Servers Repository**: https://github.com/modelcontextprotocol/servers
- **Community Examples**: https://github.com/modelcontextprotocol/servers

### HVSR Resources
- **HV-INV Repository**: https://github.com/agarcia-jerez/HV-INV
- **HVSR Method Overview**: EPA passive seismic documentation
- **Site Characterization**: USGS geophysical methods

---

## Contact and Support

**Project Maintainer**: Mersad Fathizadeh
- Email: mersadf@uark.edu
- GitHub: @mersadfathizadeh1995

**Collaborator**: Clinton Wood
- GitHub: @cmwood10
- Email: mycmwood@uark.edu

**Repository**: https://github.com/mersadfathizadeh1995/hvstrip-progressive

---

## License

This MCP server should maintain compatibility with:
- **hvstrip-progressive**: GPL-3.0 license
- **HV-INV (HVf)**: GPL-3.0 license
- **MCP SDK**: MIT license

Ensure all dependencies are properly acknowledged in LICENSE and THIRD_PARTY.md files.

---

## Final Notes

This prompt provides comprehensive guidance for creating a production-ready MCP server. Follow the phased development approach, prioritize robust error handling, and maintain clear documentation throughout. The goal is to create a tool that makes HVSR analysis accessible through AI-assisted workflows while maintaining scientific rigor and reproducibility.

Good luck with your development! 🚀
