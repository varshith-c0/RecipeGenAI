# OpenTelemetry Tracing Implementation

This document describes the OpenTelemetry (OTEL) tracing implementation for the Nosh Recipe Generation system, which enables comprehensive tracing of recipe processing requests in Arize Phoenix.

## Overview

The tracing implementation creates a single trace for each recipe processing request, capturing the complete flow from the Flask endpoint through the workflow execution to the final command generation. This provides detailed visibility into:

- Request processing time and success rates
- Individual workflow step performance
- LLM API call details and timing
- Error tracking and debugging information
- Resource utilization patterns

## Architecture

### Trace Hierarchy

Each recipe processing request creates a hierarchical trace structure:

```
recipe_processing_request (Flask endpoint)
├── recipe_processing_pipeline (Main pipeline)
│   ├── recipe_generation_workflow (Workflow execution)
│   │   ├── recipe_info_extraction (LLM extraction)
│   │   ├── ingredient_processing (Quantity conversion)
│   │   ├── cookstep_vagueness_detector (Conditional step)
│   │   │   ├── cookstep_info_extraction (LLM extraction)
│   │   │   └── cookstep_processing (Processing)
│   │   ├── vagueness_check (Validation)
│   │   └── nosh_recipe_generation (Final generation)
│   └── command_processing (Command post-processing)
```

### Key Components

1. **Tracing Utilities** (`core/tracing.py`)
   - Setup and configuration for Arize Phoenix
   - Decorators for automatic function tracing
   - Span attribute and event helpers

2. **Flask Endpoint Tracing** (`flask-app.py`)
   - Request/response details
   - YouTube processing (if applicable)
   - Validation and error handling

3. **Pipeline Tracing** (`generate_recipe.py`)
   - Two-stage pipeline (generation + command processing)
   - Stage success/failure tracking
   - Output metrics

4. **Workflow Tracing** (`core/recipe_generator.py`)
   - Complete workflow execution
   - Step-by-step progress tracking
   - Error categorization and handling

5. **Individual Step Tracing**
   - **Recipe Info Extraction** (`core/recipe_info_extr.py`)
     - LLM API calls and responses
     - Validation and error handling
     - Ingredient processing metrics
   - **Command Processing** (`core/cmd_processor.py`)
     - Post-processing steps
     - AI feature usage tracking
     - Output formatting metrics

## Configuration

### Environment Variables

```bash
# Optional: For cloud Arize Phoenix integration
PHOENIX_API_KEY=your_phoenix_api_key_here

# Default setup uses local Phoenix collector at http://localhost:6006
# This is configured in core/tracing.py
```

### Dependencies

The following packages are required for OTEL tracing:

```toml
# pyproject.toml
dependencies = [
    "arize-phoenix>=12.6.1",
    "opentelemetry-exporter-otlp>=1.37.0",
    "opentelemetry-instrumentation-flask>=0.45b0",
    "opentelemetry-instrumentation-requests>=0.45b0",
    "opentelemetry-sdk>=1.37.0",
    "openinference-instrumentation-agno>=0.1.18",
]
```

**Installation:**
```bash
# Install missing instrumentation packages
python install_tracing_deps.py

# Or install manually
pip install opentelemetry-instrumentation-flask>=0.45b0
pip install opentelemetry-instrumentation-requests>=0.45b0
```

## Usage

### Automatic Tracing

Tracing is automatically enabled when the application starts. The `setup_tracing()` function in `core/tracing.py` configures:

- Arize Phoenix connection
- Flask instrumentation
- Requests instrumentation
- Agno workflow instrumentation

### Manual Tracing

For custom functions, use the `@trace_function` decorator:

```python
from core.tracing import trace_function, add_span_attribute, add_span_event

@trace_function("custom_operation", attributes={"operation.type": "data_processing"})
def my_function(data):
    add_span_attribute("data.size", len(data))
    add_span_event("processing_started", {"data_type": type(data).__name__})
    
    # Your processing logic here
    
    add_span_event("processing_completed", {"result_size": len(result)})
    return result
```

### Span Attributes and Events

The implementation captures various attributes and events:

**Request Level:**
- `request.has_recipe_text`, `request.has_youtube_link`
- `request.ingredient_check_enabled`, `request.instruction_check_enabled`
- `request.recipe_text_length`

**Workflow Level:**
- `workflow.ingredient_check_enabled`, `workflow.instruction_check_enabled`
- `workflow.input_length`, `workflow.steps_count`
- `workflow.has_error`, `workflow.error_type`

**Step Level:**
- `extraction.input_length`, `extraction.is_recipe`
- `processing.ingredients_count`, `processing.ingredients_no_quant`
- `processor.serving_size`, `processor.consistency`

**Events:**
- `request_received`, `workflow_started`, `stage_started`
- `llm_call_completed`, `recipe_extraction_completed`
- `command_processing_completed`, `pipeline_success`

## Monitoring and Debugging

### Arize Phoenix Dashboard

**Local Setup:**
Access your traces at: `http://localhost:6006`

**Cloud Setup:**
Access your traces at: `https://app.phoenix.arize.com/s/nosh-recipe-gen`

Key metrics to monitor:

1. **Request Success Rate**
   - Filter by `response.success = true`
   - Monitor trends over time

2. **Processing Time**
   - Trace duration analysis
   - Step-by-step timing breakdown

3. **Error Analysis**
   - Filter by `workflow.has_error = true`
   - Categorize by `workflow.error_type`

4. **LLM Performance**
   - Filter by `step.type = llm_extraction`
   - Monitor `llm_call_completed` events

### Example Queries

```python
# Find failed requests
traces.where(attributes["workflow.has_error"] == True)

# Analyze processing time by recipe length
traces.where(attributes["request.recipe_text_length"] > 1000)

# Monitor LLM extraction success
traces.where(attributes["extraction.is_recipe"] == True)
```

## Testing

Run the example script to test tracing:

```bash
python example_tracing.py
```

This will:
1. Setup OTEL tracing
2. Process a sample recipe
3. Generate a complete trace in Arize Phoenix
4. Display processing results

## Troubleshooting

### Common Issues

1. **Tracing not enabled**
   - Check `PHOENIX_API_KEY` environment variable
   - Verify network connectivity to Arize Phoenix

2. **Missing traces**
   - Ensure `setup_tracing()` is called at application startup
   - Check for exceptions in the tracing setup

3. **Performance impact**
   - Tracing adds minimal overhead (~1-2ms per span)
   - Consider sampling for high-volume deployments

### Debug Mode

Enable debug logging to troubleshoot tracing issues:

```python
import logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
```

## Best Practices

1. **Span Naming**: Use descriptive, hierarchical names
2. **Attributes**: Add relevant context without sensitive data
3. **Events**: Use events for significant milestones
4. **Error Handling**: Always capture exceptions in spans
5. **Sampling**: Consider sampling for production deployments

## Future Enhancements

Potential improvements to the tracing implementation:

1. **Custom Metrics**: Add custom metrics for business KPIs
2. **Distributed Tracing**: Extend to external API calls
3. **Alerting**: Set up alerts for error rates and performance
4. **Sampling**: Implement intelligent sampling strategies
5. **Correlation**: Add correlation IDs for request tracking
