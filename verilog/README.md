# Verilog Code Guidline

The verilog backend is still at early alpha and not yet ready to use.

- Use ```my_port_name``` for variable naming.
- Always use suffix to indicate certain usage.

## Common Suffix

- ```clk```: clock
- ```rst```: reset
- ```in```: input port
- ```out```: output port
- ```en```: enable signal
- ```addr```: address port
- ```valid```: valid signal in FIFO handshake.
- ```ready```: ready signal in FIFO handshake.
