;buildInfoPackage: chisel3, version: 3.1.8, scalaVersion: 2.11.12, sbtVersion: 1.1.1, builtAtString: 2019-07-08 17:44:42.884, builtAtMillis: 1562607882884
circuit MatrixVectorMultiplication : 
  module MAC : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_2 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_3 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_4 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_5 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_6 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_7 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_8 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_9 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_10 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_11 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_12 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_13 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_14 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_15 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_2 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_3 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_4 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_5 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_6 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_7 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_8 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_9 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_10 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_11 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_12 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_13 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_14 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_1 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_2 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_3 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_4 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_5 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_6 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_7 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_8 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_9 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_10 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_11 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_12 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_13 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_14 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_15 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_1 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_2 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_3 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_4 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_5 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_6 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_7 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_8 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_9 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_10 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_11 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_12 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_13 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_14 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_16 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_17 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_18 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_19 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_20 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_21 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_22 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_23 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_24 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_25 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_26 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_27 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_28 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_29 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_30 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_31 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_15 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_16 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_17 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_18 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_19 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_20 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_21 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_22 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_23 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_24 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_25 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_26 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_27 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_28 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_29 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_16 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_17 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_18 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_19 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_20 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_21 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_22 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_23 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_24 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_25 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_26 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_27 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_28 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_29 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_30 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_31 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_15 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_16 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_17 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_18 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_19 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_20 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_21 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_22 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_23 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_24 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_25 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_26 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_27 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_28 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_29 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_32 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_33 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_34 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_35 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_36 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_37 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_38 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_39 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_40 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_41 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_42 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_43 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_44 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_45 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_46 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_47 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_30 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_31 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_32 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_33 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_34 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_35 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_36 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_37 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_38 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_39 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_40 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_41 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_42 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_43 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_44 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_2 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_32 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_33 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_34 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_35 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_36 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_37 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_38 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_39 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_40 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_41 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_42 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_43 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_44 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_45 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_46 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_47 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_30 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_31 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_32 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_33 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_34 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_35 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_36 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_37 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_38 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_39 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_40 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_41 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_42 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_43 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_44 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_48 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_49 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_50 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_51 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_52 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_53 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_54 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_55 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_56 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_57 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_58 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_59 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_60 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_61 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_62 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_63 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_45 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_46 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_47 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_48 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_49 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_50 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_51 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_52 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_53 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_54 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_55 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_56 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_57 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_58 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_59 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_3 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_48 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_49 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_50 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_51 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_52 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_53 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_54 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_55 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_56 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_57 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_58 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_59 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_60 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_61 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_62 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_63 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_45 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_46 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_47 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_48 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_49 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_50 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_51 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_52 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_53 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_54 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_55 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_56 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_57 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_58 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_59 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_64 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_65 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_66 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_67 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_68 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_69 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_70 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_71 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_72 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_73 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_74 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_75 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_76 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_77 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_78 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_79 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_60 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_61 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_62 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_63 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_64 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_65 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_66 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_67 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_68 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_69 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_70 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_71 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_72 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_73 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_74 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_4 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_64 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_65 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_66 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_67 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_68 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_69 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_70 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_71 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_72 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_73 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_74 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_75 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_76 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_77 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_78 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_79 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_60 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_61 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_62 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_63 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_64 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_65 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_66 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_67 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_68 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_69 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_70 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_71 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_72 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_73 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_74 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_80 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_81 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_82 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_83 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_84 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_85 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_86 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_87 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_88 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_89 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_90 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_91 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_92 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_93 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_94 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_95 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_75 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_76 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_77 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_78 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_79 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_80 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_81 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_82 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_83 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_84 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_85 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_86 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_87 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_88 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_89 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_5 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_80 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_81 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_82 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_83 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_84 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_85 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_86 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_87 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_88 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_89 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_90 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_91 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_92 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_93 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_94 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_95 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_75 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_76 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_77 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_78 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_79 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_80 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_81 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_82 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_83 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_84 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_85 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_86 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_87 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_88 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_89 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_96 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_97 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_98 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_99 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_100 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_101 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_102 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_103 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_104 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_105 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_106 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_107 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_108 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_109 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_110 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_111 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_90 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_91 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_92 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_93 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_94 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_95 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_96 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_97 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_98 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_99 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_100 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_101 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_102 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_103 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_104 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_6 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_96 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_97 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_98 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_99 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_100 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_101 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_102 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_103 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_104 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_105 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_106 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_107 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_108 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_109 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_110 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_111 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_90 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_91 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_92 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_93 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_94 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_95 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_96 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_97 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_98 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_99 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_100 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_101 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_102 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_103 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_104 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_112 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_113 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_114 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_115 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_116 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_117 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_118 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_119 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_120 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_121 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_122 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_123 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_124 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_125 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_126 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_127 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_105 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_106 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_107 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_108 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_109 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_110 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_111 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_112 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_113 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_114 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_115 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_116 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_117 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_118 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_119 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_7 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_112 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_113 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_114 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_115 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_116 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_117 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_118 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_119 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_120 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_121 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_122 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_123 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_124 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_125 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_126 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_127 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_105 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_106 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_107 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_108 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_109 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_110 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_111 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_112 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_113 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_114 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_115 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_116 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_117 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_118 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_119 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_128 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_129 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_130 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_131 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_132 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_133 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_134 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_135 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_136 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_137 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_138 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_139 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_140 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_141 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_142 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_143 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_120 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_121 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_122 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_123 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_124 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_125 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_126 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_127 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_128 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_129 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_130 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_131 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_132 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_133 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_134 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_8 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_128 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_129 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_130 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_131 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_132 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_133 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_134 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_135 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_136 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_137 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_138 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_139 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_140 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_141 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_142 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_143 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_120 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_121 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_122 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_123 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_124 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_125 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_126 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_127 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_128 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_129 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_130 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_131 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_132 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_133 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_134 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_144 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_145 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_146 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_147 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_148 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_149 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_150 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_151 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_152 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_153 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_154 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_155 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_156 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_157 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_158 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_159 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_135 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_136 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_137 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_138 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_139 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_140 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_141 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_142 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_143 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_144 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_145 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_146 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_147 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_148 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_149 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_9 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_144 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_145 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_146 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_147 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_148 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_149 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_150 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_151 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_152 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_153 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_154 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_155 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_156 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_157 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_158 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_159 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_135 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_136 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_137 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_138 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_139 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_140 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_141 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_142 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_143 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_144 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_145 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_146 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_147 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_148 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_149 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_160 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_161 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_162 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_163 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_164 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_165 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_166 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_167 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_168 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_169 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_170 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_171 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_172 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_173 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_174 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_175 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_150 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_151 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_152 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_153 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_154 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_155 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_156 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_157 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_158 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_159 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_160 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_161 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_162 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_163 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_164 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_10 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_160 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_161 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_162 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_163 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_164 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_165 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_166 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_167 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_168 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_169 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_170 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_171 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_172 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_173 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_174 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_175 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_150 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_151 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_152 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_153 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_154 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_155 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_156 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_157 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_158 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_159 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_160 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_161 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_162 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_163 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_164 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_176 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_177 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_178 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_179 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_180 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_181 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_182 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_183 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_184 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_185 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_186 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_187 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_188 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_189 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_190 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_191 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_165 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_166 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_167 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_168 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_169 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_170 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_171 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_172 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_173 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_174 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_175 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_176 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_177 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_178 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_179 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_11 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_176 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_177 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_178 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_179 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_180 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_181 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_182 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_183 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_184 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_185 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_186 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_187 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_188 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_189 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_190 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_191 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_165 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_166 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_167 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_168 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_169 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_170 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_171 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_172 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_173 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_174 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_175 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_176 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_177 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_178 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_179 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_192 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_193 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_194 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_195 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_196 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_197 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_198 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_199 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_200 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_201 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_202 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_203 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_204 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_205 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_206 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_207 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_180 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_181 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_182 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_183 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_184 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_185 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_186 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_187 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_188 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_189 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_190 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_191 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_192 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_193 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_194 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_12 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_192 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_193 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_194 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_195 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_196 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_197 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_198 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_199 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_200 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_201 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_202 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_203 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_204 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_205 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_206 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_207 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_180 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_181 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_182 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_183 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_184 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_185 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_186 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_187 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_188 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_189 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_190 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_191 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_192 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_193 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_194 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_208 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_209 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_210 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_211 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_212 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_213 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_214 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_215 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_216 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_217 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_218 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_219 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_220 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_221 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_222 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_223 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_195 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_196 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_197 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_198 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_199 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_200 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_201 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_202 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_203 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_204 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_205 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_206 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_207 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_208 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_209 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_13 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_208 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_209 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_210 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_211 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_212 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_213 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_214 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_215 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_216 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_217 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_218 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_219 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_220 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_221 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_222 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_223 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_195 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_196 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_197 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_198 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_199 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_200 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_201 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_202 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_203 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_204 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_205 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_206 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_207 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_208 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_209 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_224 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_225 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_226 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_227 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_228 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_229 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_230 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_231 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_232 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_233 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_234 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_235 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_236 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_237 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_238 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_239 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_210 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_211 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_212 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_213 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_214 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_215 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_216 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_217 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_218 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_219 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_220 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_221 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_222 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_223 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_224 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_14 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_224 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_225 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_226 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_227 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_228 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_229 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_230 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_231 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_232 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_233 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_234 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_235 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_236 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_237 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_238 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_239 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_210 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_211 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_212 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_213 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_214 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_215 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_216 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_217 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_218 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_219 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_220 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_221 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_222 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_223 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_224 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module MAC_240 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_241 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_242 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_243 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_244 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_245 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_246 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_247 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_248 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_249 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_250 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_251 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_252 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_253 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_254 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module MAC_255 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>, flip b : SInt<8>, flip c : SInt<1>, y : SInt<17>}
    
    wire mult : SInt<16> @[TensorGemm.scala 37:18]
    wire add : SInt<17> @[TensorGemm.scala 38:17]
    reg rA : SInt, clock @[TensorGemm.scala 39:19]
    rA <= io.a @[TensorGemm.scala 39:19]
    reg rB : SInt, clock @[TensorGemm.scala 40:19]
    rB <= io.b @[TensorGemm.scala 40:19]
    reg rC : SInt, clock @[TensorGemm.scala 41:19]
    rC <= io.c @[TensorGemm.scala 41:19]
    node _T_18 = mul(rA, rB) @[TensorGemm.scala 43:14]
    mult <= _T_18 @[TensorGemm.scala 43:8]
    node _T_19 = add(rC, mult) @[TensorGemm.scala 44:13]
    add <= _T_19 @[TensorGemm.scala 44:7]
    io.y <= add @[TensorGemm.scala 46:8]
    
  module PipeAdder_225 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_226 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_227 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_228 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_229 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_230 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_231 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_232 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<17>, flip b : SInt<17>, y : SInt<18>}
    
    wire add : SInt<18> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_233 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_234 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_235 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_236 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<18>, flip b : SInt<18>, y : SInt<19>}
    
    wire add : SInt<19> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_237 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_238 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<19>, flip b : SInt<19>, y : SInt<20>}
    
    wire add : SInt<20> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module PipeAdder_239 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<20>, flip b : SInt<20>, y : SInt<21>}
    
    wire add : SInt<21> @[TensorGemm.scala 57:17]
    reg rA : SInt, clock @[TensorGemm.scala 58:19]
    rA <= io.a @[TensorGemm.scala 58:19]
    reg rB : SInt, clock @[TensorGemm.scala 59:19]
    rB <= io.b @[TensorGemm.scala 59:19]
    node _T_14 = add(rA, rB) @[TensorGemm.scala 60:13]
    add <= _T_14 @[TensorGemm.scala 60:7]
    io.y <= add @[TensorGemm.scala 61:8]
    
  module DotProduct_15 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : SInt<8>[16], flip b : SInt<8>[16], y : SInt<21>}
    
    inst m_0 of MAC_240 @[TensorGemm.scala 77:32]
    m_0.clock <= clock
    m_0.reset <= reset
    inst m_1 of MAC_241 @[TensorGemm.scala 77:32]
    m_1.clock <= clock
    m_1.reset <= reset
    inst m_2 of MAC_242 @[TensorGemm.scala 77:32]
    m_2.clock <= clock
    m_2.reset <= reset
    inst m_3 of MAC_243 @[TensorGemm.scala 77:32]
    m_3.clock <= clock
    m_3.reset <= reset
    inst m_4 of MAC_244 @[TensorGemm.scala 77:32]
    m_4.clock <= clock
    m_4.reset <= reset
    inst m_5 of MAC_245 @[TensorGemm.scala 77:32]
    m_5.clock <= clock
    m_5.reset <= reset
    inst m_6 of MAC_246 @[TensorGemm.scala 77:32]
    m_6.clock <= clock
    m_6.reset <= reset
    inst m_7 of MAC_247 @[TensorGemm.scala 77:32]
    m_7.clock <= clock
    m_7.reset <= reset
    inst m_8 of MAC_248 @[TensorGemm.scala 77:32]
    m_8.clock <= clock
    m_8.reset <= reset
    inst m_9 of MAC_249 @[TensorGemm.scala 77:32]
    m_9.clock <= clock
    m_9.reset <= reset
    inst m_10 of MAC_250 @[TensorGemm.scala 77:32]
    m_10.clock <= clock
    m_10.reset <= reset
    inst m_11 of MAC_251 @[TensorGemm.scala 77:32]
    m_11.clock <= clock
    m_11.reset <= reset
    inst m_12 of MAC_252 @[TensorGemm.scala 77:32]
    m_12.clock <= clock
    m_12.reset <= reset
    inst m_13 of MAC_253 @[TensorGemm.scala 77:32]
    m_13.clock <= clock
    m_13.reset <= reset
    inst m_14 of MAC_254 @[TensorGemm.scala 77:32]
    m_14.clock <= clock
    m_14.reset <= reset
    inst m_15 of MAC_255 @[TensorGemm.scala 77:32]
    m_15.clock <= clock
    m_15.reset <= reset
    inst a_0_0 of PipeAdder_225 @[TensorGemm.scala 79:30]
    a_0_0.clock <= clock
    a_0_0.reset <= reset
    inst a_0_1 of PipeAdder_226 @[TensorGemm.scala 79:30]
    a_0_1.clock <= clock
    a_0_1.reset <= reset
    inst a_0_2 of PipeAdder_227 @[TensorGemm.scala 79:30]
    a_0_2.clock <= clock
    a_0_2.reset <= reset
    inst a_0_3 of PipeAdder_228 @[TensorGemm.scala 79:30]
    a_0_3.clock <= clock
    a_0_3.reset <= reset
    inst a_0_4 of PipeAdder_229 @[TensorGemm.scala 79:30]
    a_0_4.clock <= clock
    a_0_4.reset <= reset
    inst a_0_5 of PipeAdder_230 @[TensorGemm.scala 79:30]
    a_0_5.clock <= clock
    a_0_5.reset <= reset
    inst a_0_6 of PipeAdder_231 @[TensorGemm.scala 79:30]
    a_0_6.clock <= clock
    a_0_6.reset <= reset
    inst a_0_7 of PipeAdder_232 @[TensorGemm.scala 79:30]
    a_0_7.clock <= clock
    a_0_7.reset <= reset
    inst a_1_0 of PipeAdder_233 @[TensorGemm.scala 79:30]
    a_1_0.clock <= clock
    a_1_0.reset <= reset
    inst a_1_1 of PipeAdder_234 @[TensorGemm.scala 79:30]
    a_1_1.clock <= clock
    a_1_1.reset <= reset
    inst a_1_2 of PipeAdder_235 @[TensorGemm.scala 79:30]
    a_1_2.clock <= clock
    a_1_2.reset <= reset
    inst a_1_3 of PipeAdder_236 @[TensorGemm.scala 79:30]
    a_1_3.clock <= clock
    a_1_3.reset <= reset
    inst a_2_0 of PipeAdder_237 @[TensorGemm.scala 79:30]
    a_2_0.clock <= clock
    a_2_0.reset <= reset
    inst a_2_1 of PipeAdder_238 @[TensorGemm.scala 79:30]
    a_2_1.clock <= clock
    a_2_1.reset <= reset
    inst a_3_0 of PipeAdder_239 @[TensorGemm.scala 79:30]
    a_3_0.clock <= clock
    a_3_0.reset <= reset
    m_0.io.a <= io.a[0] @[TensorGemm.scala 84:15]
    m_0.io.b <= io.b[0] @[TensorGemm.scala 85:15]
    m_0.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_1.io.a <= io.a[1] @[TensorGemm.scala 84:15]
    m_1.io.b <= io.b[1] @[TensorGemm.scala 85:15]
    m_1.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_2.io.a <= io.a[2] @[TensorGemm.scala 84:15]
    m_2.io.b <= io.b[2] @[TensorGemm.scala 85:15]
    m_2.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_3.io.a <= io.a[3] @[TensorGemm.scala 84:15]
    m_3.io.b <= io.b[3] @[TensorGemm.scala 85:15]
    m_3.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_4.io.a <= io.a[4] @[TensorGemm.scala 84:15]
    m_4.io.b <= io.b[4] @[TensorGemm.scala 85:15]
    m_4.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_5.io.a <= io.a[5] @[TensorGemm.scala 84:15]
    m_5.io.b <= io.b[5] @[TensorGemm.scala 85:15]
    m_5.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_6.io.a <= io.a[6] @[TensorGemm.scala 84:15]
    m_6.io.b <= io.b[6] @[TensorGemm.scala 85:15]
    m_6.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_7.io.a <= io.a[7] @[TensorGemm.scala 84:15]
    m_7.io.b <= io.b[7] @[TensorGemm.scala 85:15]
    m_7.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_8.io.a <= io.a[8] @[TensorGemm.scala 84:15]
    m_8.io.b <= io.b[8] @[TensorGemm.scala 85:15]
    m_8.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_9.io.a <= io.a[9] @[TensorGemm.scala 84:15]
    m_9.io.b <= io.b[9] @[TensorGemm.scala 85:15]
    m_9.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_10.io.a <= io.a[10] @[TensorGemm.scala 84:15]
    m_10.io.b <= io.b[10] @[TensorGemm.scala 85:15]
    m_10.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_11.io.a <= io.a[11] @[TensorGemm.scala 84:15]
    m_11.io.b <= io.b[11] @[TensorGemm.scala 85:15]
    m_11.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_12.io.a <= io.a[12] @[TensorGemm.scala 84:15]
    m_12.io.b <= io.b[12] @[TensorGemm.scala 85:15]
    m_12.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_13.io.a <= io.a[13] @[TensorGemm.scala 84:15]
    m_13.io.b <= io.b[13] @[TensorGemm.scala 85:15]
    m_13.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_14.io.a <= io.a[14] @[TensorGemm.scala 84:15]
    m_14.io.b <= io.b[14] @[TensorGemm.scala 85:15]
    m_14.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    m_15.io.a <= io.a[15] @[TensorGemm.scala 84:15]
    m_15.io.b <= io.b[15] @[TensorGemm.scala 85:15]
    m_15.io.c <= asSInt(UInt<1>("h00")) @[TensorGemm.scala 86:15]
    a_0_0.io.a <= m_0.io.y @[TensorGemm.scala 94:22]
    a_0_0.io.b <= m_1.io.y @[TensorGemm.scala 95:22]
    a_0_1.io.a <= m_2.io.y @[TensorGemm.scala 94:22]
    a_0_1.io.b <= m_3.io.y @[TensorGemm.scala 95:22]
    a_0_2.io.a <= m_4.io.y @[TensorGemm.scala 94:22]
    a_0_2.io.b <= m_5.io.y @[TensorGemm.scala 95:22]
    a_0_3.io.a <= m_6.io.y @[TensorGemm.scala 94:22]
    a_0_3.io.b <= m_7.io.y @[TensorGemm.scala 95:22]
    a_0_4.io.a <= m_8.io.y @[TensorGemm.scala 94:22]
    a_0_4.io.b <= m_9.io.y @[TensorGemm.scala 95:22]
    a_0_5.io.a <= m_10.io.y @[TensorGemm.scala 94:22]
    a_0_5.io.b <= m_11.io.y @[TensorGemm.scala 95:22]
    a_0_6.io.a <= m_12.io.y @[TensorGemm.scala 94:22]
    a_0_6.io.b <= m_13.io.y @[TensorGemm.scala 95:22]
    a_0_7.io.a <= m_14.io.y @[TensorGemm.scala 94:22]
    a_0_7.io.b <= m_15.io.y @[TensorGemm.scala 95:22]
    a_1_0.io.a <= a_0_0.io.y @[TensorGemm.scala 97:22]
    a_1_0.io.b <= a_0_1.io.y @[TensorGemm.scala 98:22]
    a_1_1.io.a <= a_0_2.io.y @[TensorGemm.scala 97:22]
    a_1_1.io.b <= a_0_3.io.y @[TensorGemm.scala 98:22]
    a_1_2.io.a <= a_0_4.io.y @[TensorGemm.scala 97:22]
    a_1_2.io.b <= a_0_5.io.y @[TensorGemm.scala 98:22]
    a_1_3.io.a <= a_0_6.io.y @[TensorGemm.scala 97:22]
    a_1_3.io.b <= a_0_7.io.y @[TensorGemm.scala 98:22]
    a_2_0.io.a <= a_1_0.io.y @[TensorGemm.scala 97:22]
    a_2_0.io.b <= a_1_1.io.y @[TensorGemm.scala 98:22]
    a_2_1.io.a <= a_1_2.io.y @[TensorGemm.scala 97:22]
    a_2_1.io.b <= a_1_3.io.y @[TensorGemm.scala 98:22]
    a_3_0.io.a <= a_2_0.io.y @[TensorGemm.scala 97:22]
    a_3_0.io.b <= a_2_1.io.y @[TensorGemm.scala 98:22]
    io.y <= a_3_0.io.y @[TensorGemm.scala 104:8]
    
  module Pipe : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_2 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_3 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_4 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_5 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_6 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_7 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_8 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_9 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_10 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_11 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_12 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_13 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_14 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module Pipe_15 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {valid : UInt<1>, bits : UInt<32>}, deq : {valid : UInt<1>, bits : UInt<32>}}
    
    reg _T_19 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_19 <= io.enq.valid @[Valid.scala 48:22]
    reg _T_21 : UInt<32>, clock @[Reg.scala 11:16]
    when io.enq.valid : @[Reg.scala 12:19]
      _T_21 <= io.enq.bits @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_24 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_24 <= _T_19 @[Valid.scala 48:22]
    reg _T_26 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_19 : @[Reg.scala 12:19]
      _T_26 <= _T_21 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_29 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_29 <= _T_24 @[Valid.scala 48:22]
    reg _T_31 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_24 : @[Reg.scala 12:19]
      _T_31 <= _T_26 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_34 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_34 <= _T_29 @[Valid.scala 48:22]
    reg _T_36 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_29 : @[Reg.scala 12:19]
      _T_36 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_39 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Valid.scala 48:22]
    _T_39 <= _T_34 @[Valid.scala 48:22]
    reg _T_41 : UInt<32>, clock @[Reg.scala 11:16]
    when _T_34 : @[Reg.scala 12:19]
      _T_41 <= _T_36 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    wire _T_45 : {valid : UInt<1>, bits : UInt<32>} @[Valid.scala 43:21]
    _T_45.valid <= _T_39 @[Valid.scala 44:17]
    _T_45.bits <= _T_41 @[Valid.scala 45:16]
    io.deq.bits <= _T_45.bits @[Valid.scala 70:10]
    io.deq.valid <= _T_45.valid @[Valid.scala 70:10]
    
  module MatrixVectorMultiplication : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip reset : UInt<1>, inp : {flip data : {valid : UInt<1>, bits : UInt<8>[16][1]}}, wgt : {flip data : {valid : UInt<1>, bits : UInt<8>[16][16]}}, acc_i : {flip data : {valid : UInt<1>, bits : UInt<32>[16][1]}}, acc_o : {data : {valid : UInt<1>, bits : UInt<32>[16][1]}}, out : {data : {valid : UInt<1>, bits : UInt<8>[16][1]}}}
    
    inst dot_0 of DotProduct @[TensorGemm.scala 122:34]
    dot_0.clock <= clock
    dot_0.reset <= reset
    inst dot_1 of DotProduct_1 @[TensorGemm.scala 122:34]
    dot_1.clock <= clock
    dot_1.reset <= reset
    inst dot_2 of DotProduct_2 @[TensorGemm.scala 122:34]
    dot_2.clock <= clock
    dot_2.reset <= reset
    inst dot_3 of DotProduct_3 @[TensorGemm.scala 122:34]
    dot_3.clock <= clock
    dot_3.reset <= reset
    inst dot_4 of DotProduct_4 @[TensorGemm.scala 122:34]
    dot_4.clock <= clock
    dot_4.reset <= reset
    inst dot_5 of DotProduct_5 @[TensorGemm.scala 122:34]
    dot_5.clock <= clock
    dot_5.reset <= reset
    inst dot_6 of DotProduct_6 @[TensorGemm.scala 122:34]
    dot_6.clock <= clock
    dot_6.reset <= reset
    inst dot_7 of DotProduct_7 @[TensorGemm.scala 122:34]
    dot_7.clock <= clock
    dot_7.reset <= reset
    inst dot_8 of DotProduct_8 @[TensorGemm.scala 122:34]
    dot_8.clock <= clock
    dot_8.reset <= reset
    inst dot_9 of DotProduct_9 @[TensorGemm.scala 122:34]
    dot_9.clock <= clock
    dot_9.reset <= reset
    inst dot_10 of DotProduct_10 @[TensorGemm.scala 122:34]
    dot_10.clock <= clock
    dot_10.reset <= reset
    inst dot_11 of DotProduct_11 @[TensorGemm.scala 122:34]
    dot_11.clock <= clock
    dot_11.reset <= reset
    inst dot_12 of DotProduct_12 @[TensorGemm.scala 122:34]
    dot_12.clock <= clock
    dot_12.reset <= reset
    inst dot_13 of DotProduct_13 @[TensorGemm.scala 122:34]
    dot_13.clock <= clock
    dot_13.reset <= reset
    inst dot_14 of DotProduct_14 @[TensorGemm.scala 122:34]
    dot_14.clock <= clock
    dot_14.reset <= reset
    inst dot_15 of DotProduct_15 @[TensorGemm.scala 122:34]
    dot_15.clock <= clock
    dot_15.reset <= reset
    inst acc_0 of Pipe @[TensorGemm.scala 123:34]
    acc_0.clock <= clock
    acc_0.reset <= reset
    inst acc_1 of Pipe_1 @[TensorGemm.scala 123:34]
    acc_1.clock <= clock
    acc_1.reset <= reset
    inst acc_2 of Pipe_2 @[TensorGemm.scala 123:34]
    acc_2.clock <= clock
    acc_2.reset <= reset
    inst acc_3 of Pipe_3 @[TensorGemm.scala 123:34]
    acc_3.clock <= clock
    acc_3.reset <= reset
    inst acc_4 of Pipe_4 @[TensorGemm.scala 123:34]
    acc_4.clock <= clock
    acc_4.reset <= reset
    inst acc_5 of Pipe_5 @[TensorGemm.scala 123:34]
    acc_5.clock <= clock
    acc_5.reset <= reset
    inst acc_6 of Pipe_6 @[TensorGemm.scala 123:34]
    acc_6.clock <= clock
    acc_6.reset <= reset
    inst acc_7 of Pipe_7 @[TensorGemm.scala 123:34]
    acc_7.clock <= clock
    acc_7.reset <= reset
    inst acc_8 of Pipe_8 @[TensorGemm.scala 123:34]
    acc_8.clock <= clock
    acc_8.reset <= reset
    inst acc_9 of Pipe_9 @[TensorGemm.scala 123:34]
    acc_9.clock <= clock
    acc_9.reset <= reset
    inst acc_10 of Pipe_10 @[TensorGemm.scala 123:34]
    acc_10.clock <= clock
    acc_10.reset <= reset
    inst acc_11 of Pipe_11 @[TensorGemm.scala 123:34]
    acc_11.clock <= clock
    acc_11.reset <= reset
    inst acc_12 of Pipe_12 @[TensorGemm.scala 123:34]
    acc_12.clock <= clock
    acc_12.reset <= reset
    inst acc_13 of Pipe_13 @[TensorGemm.scala 123:34]
    acc_13.clock <= clock
    acc_13.reset <= reset
    inst acc_14 of Pipe_14 @[TensorGemm.scala 123:34]
    acc_14.clock <= clock
    acc_14.reset <= reset
    inst acc_15 of Pipe_15 @[TensorGemm.scala 123:34]
    acc_15.clock <= clock
    acc_15.reset <= reset
    wire add_0 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_1 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_2 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_3 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_4 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_5 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_6 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_7 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_8 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_9 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_10 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_11 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_12 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_13 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_14 : SInt<32> @[TensorGemm.scala 124:32]
    wire add_15 : SInt<32> @[TensorGemm.scala 124:32]
    wire vld : UInt<1>[16] @[TensorGemm.scala 125:17]
    node _T_6016 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6017 = and(_T_6016, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6018 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6019 = and(_T_6017, _T_6018) @[TensorGemm.scala 128:88]
    acc_0.io.enq.valid <= _T_6019 @[TensorGemm.scala 128:25]
    acc_0.io.enq.bits <= io.acc_i.data.bits[0][0] @[TensorGemm.scala 129:24]
    node _T_6020 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_0.io.a[0] <= _T_6020 @[TensorGemm.scala 131:22]
    node _T_6021 = asSInt(io.wgt.data.bits[0][0]) @[TensorGemm.scala 132:48]
    dot_0.io.b[0] <= _T_6021 @[TensorGemm.scala 132:22]
    node _T_6022 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_0.io.a[1] <= _T_6022 @[TensorGemm.scala 131:22]
    node _T_6023 = asSInt(io.wgt.data.bits[0][1]) @[TensorGemm.scala 132:48]
    dot_0.io.b[1] <= _T_6023 @[TensorGemm.scala 132:22]
    node _T_6024 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_0.io.a[2] <= _T_6024 @[TensorGemm.scala 131:22]
    node _T_6025 = asSInt(io.wgt.data.bits[0][2]) @[TensorGemm.scala 132:48]
    dot_0.io.b[2] <= _T_6025 @[TensorGemm.scala 132:22]
    node _T_6026 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_0.io.a[3] <= _T_6026 @[TensorGemm.scala 131:22]
    node _T_6027 = asSInt(io.wgt.data.bits[0][3]) @[TensorGemm.scala 132:48]
    dot_0.io.b[3] <= _T_6027 @[TensorGemm.scala 132:22]
    node _T_6028 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_0.io.a[4] <= _T_6028 @[TensorGemm.scala 131:22]
    node _T_6029 = asSInt(io.wgt.data.bits[0][4]) @[TensorGemm.scala 132:48]
    dot_0.io.b[4] <= _T_6029 @[TensorGemm.scala 132:22]
    node _T_6030 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_0.io.a[5] <= _T_6030 @[TensorGemm.scala 131:22]
    node _T_6031 = asSInt(io.wgt.data.bits[0][5]) @[TensorGemm.scala 132:48]
    dot_0.io.b[5] <= _T_6031 @[TensorGemm.scala 132:22]
    node _T_6032 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_0.io.a[6] <= _T_6032 @[TensorGemm.scala 131:22]
    node _T_6033 = asSInt(io.wgt.data.bits[0][6]) @[TensorGemm.scala 132:48]
    dot_0.io.b[6] <= _T_6033 @[TensorGemm.scala 132:22]
    node _T_6034 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_0.io.a[7] <= _T_6034 @[TensorGemm.scala 131:22]
    node _T_6035 = asSInt(io.wgt.data.bits[0][7]) @[TensorGemm.scala 132:48]
    dot_0.io.b[7] <= _T_6035 @[TensorGemm.scala 132:22]
    node _T_6036 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_0.io.a[8] <= _T_6036 @[TensorGemm.scala 131:22]
    node _T_6037 = asSInt(io.wgt.data.bits[0][8]) @[TensorGemm.scala 132:48]
    dot_0.io.b[8] <= _T_6037 @[TensorGemm.scala 132:22]
    node _T_6038 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_0.io.a[9] <= _T_6038 @[TensorGemm.scala 131:22]
    node _T_6039 = asSInt(io.wgt.data.bits[0][9]) @[TensorGemm.scala 132:48]
    dot_0.io.b[9] <= _T_6039 @[TensorGemm.scala 132:22]
    node _T_6040 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_0.io.a[10] <= _T_6040 @[TensorGemm.scala 131:22]
    node _T_6041 = asSInt(io.wgt.data.bits[0][10]) @[TensorGemm.scala 132:48]
    dot_0.io.b[10] <= _T_6041 @[TensorGemm.scala 132:22]
    node _T_6042 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_0.io.a[11] <= _T_6042 @[TensorGemm.scala 131:22]
    node _T_6043 = asSInt(io.wgt.data.bits[0][11]) @[TensorGemm.scala 132:48]
    dot_0.io.b[11] <= _T_6043 @[TensorGemm.scala 132:22]
    node _T_6044 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_0.io.a[12] <= _T_6044 @[TensorGemm.scala 131:22]
    node _T_6045 = asSInt(io.wgt.data.bits[0][12]) @[TensorGemm.scala 132:48]
    dot_0.io.b[12] <= _T_6045 @[TensorGemm.scala 132:22]
    node _T_6046 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_0.io.a[13] <= _T_6046 @[TensorGemm.scala 131:22]
    node _T_6047 = asSInt(io.wgt.data.bits[0][13]) @[TensorGemm.scala 132:48]
    dot_0.io.b[13] <= _T_6047 @[TensorGemm.scala 132:22]
    node _T_6048 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_0.io.a[14] <= _T_6048 @[TensorGemm.scala 131:22]
    node _T_6049 = asSInt(io.wgt.data.bits[0][14]) @[TensorGemm.scala 132:48]
    dot_0.io.b[14] <= _T_6049 @[TensorGemm.scala 132:22]
    node _T_6050 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_0.io.a[15] <= _T_6050 @[TensorGemm.scala 131:22]
    node _T_6051 = asSInt(io.wgt.data.bits[0][15]) @[TensorGemm.scala 132:48]
    dot_0.io.b[15] <= _T_6051 @[TensorGemm.scala 132:22]
    node _T_6052 = asSInt(acc_0.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6053 = add(_T_6052, dot_0.io.y) @[TensorGemm.scala 134:41]
    node _T_6054 = tail(_T_6053, 1) @[TensorGemm.scala 134:41]
    node _T_6055 = asSInt(_T_6054) @[TensorGemm.scala 134:41]
    add_0 <= _T_6055 @[TensorGemm.scala 134:12]
    node _T_6057 = asUInt(add_0) @[TensorGemm.scala 135:59]
    node _T_6058 = mux(io.reset, UInt<1>("h00"), _T_6057) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][0] <= _T_6058 @[TensorGemm.scala 135:30]
    node _T_6059 = asUInt(add_0) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][0] <= _T_6059 @[TensorGemm.scala 136:28]
    vld[0] <= acc_0.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6060 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6061 = and(_T_6060, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6062 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6063 = and(_T_6061, _T_6062) @[TensorGemm.scala 128:88]
    acc_1.io.enq.valid <= _T_6063 @[TensorGemm.scala 128:25]
    acc_1.io.enq.bits <= io.acc_i.data.bits[0][1] @[TensorGemm.scala 129:24]
    node _T_6064 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_1.io.a[0] <= _T_6064 @[TensorGemm.scala 131:22]
    node _T_6065 = asSInt(io.wgt.data.bits[1][0]) @[TensorGemm.scala 132:48]
    dot_1.io.b[0] <= _T_6065 @[TensorGemm.scala 132:22]
    node _T_6066 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_1.io.a[1] <= _T_6066 @[TensorGemm.scala 131:22]
    node _T_6067 = asSInt(io.wgt.data.bits[1][1]) @[TensorGemm.scala 132:48]
    dot_1.io.b[1] <= _T_6067 @[TensorGemm.scala 132:22]
    node _T_6068 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_1.io.a[2] <= _T_6068 @[TensorGemm.scala 131:22]
    node _T_6069 = asSInt(io.wgt.data.bits[1][2]) @[TensorGemm.scala 132:48]
    dot_1.io.b[2] <= _T_6069 @[TensorGemm.scala 132:22]
    node _T_6070 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_1.io.a[3] <= _T_6070 @[TensorGemm.scala 131:22]
    node _T_6071 = asSInt(io.wgt.data.bits[1][3]) @[TensorGemm.scala 132:48]
    dot_1.io.b[3] <= _T_6071 @[TensorGemm.scala 132:22]
    node _T_6072 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_1.io.a[4] <= _T_6072 @[TensorGemm.scala 131:22]
    node _T_6073 = asSInt(io.wgt.data.bits[1][4]) @[TensorGemm.scala 132:48]
    dot_1.io.b[4] <= _T_6073 @[TensorGemm.scala 132:22]
    node _T_6074 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_1.io.a[5] <= _T_6074 @[TensorGemm.scala 131:22]
    node _T_6075 = asSInt(io.wgt.data.bits[1][5]) @[TensorGemm.scala 132:48]
    dot_1.io.b[5] <= _T_6075 @[TensorGemm.scala 132:22]
    node _T_6076 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_1.io.a[6] <= _T_6076 @[TensorGemm.scala 131:22]
    node _T_6077 = asSInt(io.wgt.data.bits[1][6]) @[TensorGemm.scala 132:48]
    dot_1.io.b[6] <= _T_6077 @[TensorGemm.scala 132:22]
    node _T_6078 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_1.io.a[7] <= _T_6078 @[TensorGemm.scala 131:22]
    node _T_6079 = asSInt(io.wgt.data.bits[1][7]) @[TensorGemm.scala 132:48]
    dot_1.io.b[7] <= _T_6079 @[TensorGemm.scala 132:22]
    node _T_6080 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_1.io.a[8] <= _T_6080 @[TensorGemm.scala 131:22]
    node _T_6081 = asSInt(io.wgt.data.bits[1][8]) @[TensorGemm.scala 132:48]
    dot_1.io.b[8] <= _T_6081 @[TensorGemm.scala 132:22]
    node _T_6082 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_1.io.a[9] <= _T_6082 @[TensorGemm.scala 131:22]
    node _T_6083 = asSInt(io.wgt.data.bits[1][9]) @[TensorGemm.scala 132:48]
    dot_1.io.b[9] <= _T_6083 @[TensorGemm.scala 132:22]
    node _T_6084 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_1.io.a[10] <= _T_6084 @[TensorGemm.scala 131:22]
    node _T_6085 = asSInt(io.wgt.data.bits[1][10]) @[TensorGemm.scala 132:48]
    dot_1.io.b[10] <= _T_6085 @[TensorGemm.scala 132:22]
    node _T_6086 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_1.io.a[11] <= _T_6086 @[TensorGemm.scala 131:22]
    node _T_6087 = asSInt(io.wgt.data.bits[1][11]) @[TensorGemm.scala 132:48]
    dot_1.io.b[11] <= _T_6087 @[TensorGemm.scala 132:22]
    node _T_6088 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_1.io.a[12] <= _T_6088 @[TensorGemm.scala 131:22]
    node _T_6089 = asSInt(io.wgt.data.bits[1][12]) @[TensorGemm.scala 132:48]
    dot_1.io.b[12] <= _T_6089 @[TensorGemm.scala 132:22]
    node _T_6090 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_1.io.a[13] <= _T_6090 @[TensorGemm.scala 131:22]
    node _T_6091 = asSInt(io.wgt.data.bits[1][13]) @[TensorGemm.scala 132:48]
    dot_1.io.b[13] <= _T_6091 @[TensorGemm.scala 132:22]
    node _T_6092 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_1.io.a[14] <= _T_6092 @[TensorGemm.scala 131:22]
    node _T_6093 = asSInt(io.wgt.data.bits[1][14]) @[TensorGemm.scala 132:48]
    dot_1.io.b[14] <= _T_6093 @[TensorGemm.scala 132:22]
    node _T_6094 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_1.io.a[15] <= _T_6094 @[TensorGemm.scala 131:22]
    node _T_6095 = asSInt(io.wgt.data.bits[1][15]) @[TensorGemm.scala 132:48]
    dot_1.io.b[15] <= _T_6095 @[TensorGemm.scala 132:22]
    node _T_6096 = asSInt(acc_1.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6097 = add(_T_6096, dot_1.io.y) @[TensorGemm.scala 134:41]
    node _T_6098 = tail(_T_6097, 1) @[TensorGemm.scala 134:41]
    node _T_6099 = asSInt(_T_6098) @[TensorGemm.scala 134:41]
    add_1 <= _T_6099 @[TensorGemm.scala 134:12]
    node _T_6101 = asUInt(add_1) @[TensorGemm.scala 135:59]
    node _T_6102 = mux(io.reset, UInt<1>("h00"), _T_6101) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][1] <= _T_6102 @[TensorGemm.scala 135:30]
    node _T_6103 = asUInt(add_1) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][1] <= _T_6103 @[TensorGemm.scala 136:28]
    vld[1] <= acc_1.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6104 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6105 = and(_T_6104, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6106 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6107 = and(_T_6105, _T_6106) @[TensorGemm.scala 128:88]
    acc_2.io.enq.valid <= _T_6107 @[TensorGemm.scala 128:25]
    acc_2.io.enq.bits <= io.acc_i.data.bits[0][2] @[TensorGemm.scala 129:24]
    node _T_6108 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_2.io.a[0] <= _T_6108 @[TensorGemm.scala 131:22]
    node _T_6109 = asSInt(io.wgt.data.bits[2][0]) @[TensorGemm.scala 132:48]
    dot_2.io.b[0] <= _T_6109 @[TensorGemm.scala 132:22]
    node _T_6110 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_2.io.a[1] <= _T_6110 @[TensorGemm.scala 131:22]
    node _T_6111 = asSInt(io.wgt.data.bits[2][1]) @[TensorGemm.scala 132:48]
    dot_2.io.b[1] <= _T_6111 @[TensorGemm.scala 132:22]
    node _T_6112 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_2.io.a[2] <= _T_6112 @[TensorGemm.scala 131:22]
    node _T_6113 = asSInt(io.wgt.data.bits[2][2]) @[TensorGemm.scala 132:48]
    dot_2.io.b[2] <= _T_6113 @[TensorGemm.scala 132:22]
    node _T_6114 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_2.io.a[3] <= _T_6114 @[TensorGemm.scala 131:22]
    node _T_6115 = asSInt(io.wgt.data.bits[2][3]) @[TensorGemm.scala 132:48]
    dot_2.io.b[3] <= _T_6115 @[TensorGemm.scala 132:22]
    node _T_6116 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_2.io.a[4] <= _T_6116 @[TensorGemm.scala 131:22]
    node _T_6117 = asSInt(io.wgt.data.bits[2][4]) @[TensorGemm.scala 132:48]
    dot_2.io.b[4] <= _T_6117 @[TensorGemm.scala 132:22]
    node _T_6118 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_2.io.a[5] <= _T_6118 @[TensorGemm.scala 131:22]
    node _T_6119 = asSInt(io.wgt.data.bits[2][5]) @[TensorGemm.scala 132:48]
    dot_2.io.b[5] <= _T_6119 @[TensorGemm.scala 132:22]
    node _T_6120 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_2.io.a[6] <= _T_6120 @[TensorGemm.scala 131:22]
    node _T_6121 = asSInt(io.wgt.data.bits[2][6]) @[TensorGemm.scala 132:48]
    dot_2.io.b[6] <= _T_6121 @[TensorGemm.scala 132:22]
    node _T_6122 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_2.io.a[7] <= _T_6122 @[TensorGemm.scala 131:22]
    node _T_6123 = asSInt(io.wgt.data.bits[2][7]) @[TensorGemm.scala 132:48]
    dot_2.io.b[7] <= _T_6123 @[TensorGemm.scala 132:22]
    node _T_6124 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_2.io.a[8] <= _T_6124 @[TensorGemm.scala 131:22]
    node _T_6125 = asSInt(io.wgt.data.bits[2][8]) @[TensorGemm.scala 132:48]
    dot_2.io.b[8] <= _T_6125 @[TensorGemm.scala 132:22]
    node _T_6126 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_2.io.a[9] <= _T_6126 @[TensorGemm.scala 131:22]
    node _T_6127 = asSInt(io.wgt.data.bits[2][9]) @[TensorGemm.scala 132:48]
    dot_2.io.b[9] <= _T_6127 @[TensorGemm.scala 132:22]
    node _T_6128 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_2.io.a[10] <= _T_6128 @[TensorGemm.scala 131:22]
    node _T_6129 = asSInt(io.wgt.data.bits[2][10]) @[TensorGemm.scala 132:48]
    dot_2.io.b[10] <= _T_6129 @[TensorGemm.scala 132:22]
    node _T_6130 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_2.io.a[11] <= _T_6130 @[TensorGemm.scala 131:22]
    node _T_6131 = asSInt(io.wgt.data.bits[2][11]) @[TensorGemm.scala 132:48]
    dot_2.io.b[11] <= _T_6131 @[TensorGemm.scala 132:22]
    node _T_6132 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_2.io.a[12] <= _T_6132 @[TensorGemm.scala 131:22]
    node _T_6133 = asSInt(io.wgt.data.bits[2][12]) @[TensorGemm.scala 132:48]
    dot_2.io.b[12] <= _T_6133 @[TensorGemm.scala 132:22]
    node _T_6134 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_2.io.a[13] <= _T_6134 @[TensorGemm.scala 131:22]
    node _T_6135 = asSInt(io.wgt.data.bits[2][13]) @[TensorGemm.scala 132:48]
    dot_2.io.b[13] <= _T_6135 @[TensorGemm.scala 132:22]
    node _T_6136 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_2.io.a[14] <= _T_6136 @[TensorGemm.scala 131:22]
    node _T_6137 = asSInt(io.wgt.data.bits[2][14]) @[TensorGemm.scala 132:48]
    dot_2.io.b[14] <= _T_6137 @[TensorGemm.scala 132:22]
    node _T_6138 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_2.io.a[15] <= _T_6138 @[TensorGemm.scala 131:22]
    node _T_6139 = asSInt(io.wgt.data.bits[2][15]) @[TensorGemm.scala 132:48]
    dot_2.io.b[15] <= _T_6139 @[TensorGemm.scala 132:22]
    node _T_6140 = asSInt(acc_2.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6141 = add(_T_6140, dot_2.io.y) @[TensorGemm.scala 134:41]
    node _T_6142 = tail(_T_6141, 1) @[TensorGemm.scala 134:41]
    node _T_6143 = asSInt(_T_6142) @[TensorGemm.scala 134:41]
    add_2 <= _T_6143 @[TensorGemm.scala 134:12]
    node _T_6145 = asUInt(add_2) @[TensorGemm.scala 135:59]
    node _T_6146 = mux(io.reset, UInt<1>("h00"), _T_6145) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][2] <= _T_6146 @[TensorGemm.scala 135:30]
    node _T_6147 = asUInt(add_2) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][2] <= _T_6147 @[TensorGemm.scala 136:28]
    vld[2] <= acc_2.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6148 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6149 = and(_T_6148, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6150 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6151 = and(_T_6149, _T_6150) @[TensorGemm.scala 128:88]
    acc_3.io.enq.valid <= _T_6151 @[TensorGemm.scala 128:25]
    acc_3.io.enq.bits <= io.acc_i.data.bits[0][3] @[TensorGemm.scala 129:24]
    node _T_6152 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_3.io.a[0] <= _T_6152 @[TensorGemm.scala 131:22]
    node _T_6153 = asSInt(io.wgt.data.bits[3][0]) @[TensorGemm.scala 132:48]
    dot_3.io.b[0] <= _T_6153 @[TensorGemm.scala 132:22]
    node _T_6154 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_3.io.a[1] <= _T_6154 @[TensorGemm.scala 131:22]
    node _T_6155 = asSInt(io.wgt.data.bits[3][1]) @[TensorGemm.scala 132:48]
    dot_3.io.b[1] <= _T_6155 @[TensorGemm.scala 132:22]
    node _T_6156 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_3.io.a[2] <= _T_6156 @[TensorGemm.scala 131:22]
    node _T_6157 = asSInt(io.wgt.data.bits[3][2]) @[TensorGemm.scala 132:48]
    dot_3.io.b[2] <= _T_6157 @[TensorGemm.scala 132:22]
    node _T_6158 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_3.io.a[3] <= _T_6158 @[TensorGemm.scala 131:22]
    node _T_6159 = asSInt(io.wgt.data.bits[3][3]) @[TensorGemm.scala 132:48]
    dot_3.io.b[3] <= _T_6159 @[TensorGemm.scala 132:22]
    node _T_6160 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_3.io.a[4] <= _T_6160 @[TensorGemm.scala 131:22]
    node _T_6161 = asSInt(io.wgt.data.bits[3][4]) @[TensorGemm.scala 132:48]
    dot_3.io.b[4] <= _T_6161 @[TensorGemm.scala 132:22]
    node _T_6162 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_3.io.a[5] <= _T_6162 @[TensorGemm.scala 131:22]
    node _T_6163 = asSInt(io.wgt.data.bits[3][5]) @[TensorGemm.scala 132:48]
    dot_3.io.b[5] <= _T_6163 @[TensorGemm.scala 132:22]
    node _T_6164 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_3.io.a[6] <= _T_6164 @[TensorGemm.scala 131:22]
    node _T_6165 = asSInt(io.wgt.data.bits[3][6]) @[TensorGemm.scala 132:48]
    dot_3.io.b[6] <= _T_6165 @[TensorGemm.scala 132:22]
    node _T_6166 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_3.io.a[7] <= _T_6166 @[TensorGemm.scala 131:22]
    node _T_6167 = asSInt(io.wgt.data.bits[3][7]) @[TensorGemm.scala 132:48]
    dot_3.io.b[7] <= _T_6167 @[TensorGemm.scala 132:22]
    node _T_6168 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_3.io.a[8] <= _T_6168 @[TensorGemm.scala 131:22]
    node _T_6169 = asSInt(io.wgt.data.bits[3][8]) @[TensorGemm.scala 132:48]
    dot_3.io.b[8] <= _T_6169 @[TensorGemm.scala 132:22]
    node _T_6170 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_3.io.a[9] <= _T_6170 @[TensorGemm.scala 131:22]
    node _T_6171 = asSInt(io.wgt.data.bits[3][9]) @[TensorGemm.scala 132:48]
    dot_3.io.b[9] <= _T_6171 @[TensorGemm.scala 132:22]
    node _T_6172 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_3.io.a[10] <= _T_6172 @[TensorGemm.scala 131:22]
    node _T_6173 = asSInt(io.wgt.data.bits[3][10]) @[TensorGemm.scala 132:48]
    dot_3.io.b[10] <= _T_6173 @[TensorGemm.scala 132:22]
    node _T_6174 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_3.io.a[11] <= _T_6174 @[TensorGemm.scala 131:22]
    node _T_6175 = asSInt(io.wgt.data.bits[3][11]) @[TensorGemm.scala 132:48]
    dot_3.io.b[11] <= _T_6175 @[TensorGemm.scala 132:22]
    node _T_6176 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_3.io.a[12] <= _T_6176 @[TensorGemm.scala 131:22]
    node _T_6177 = asSInt(io.wgt.data.bits[3][12]) @[TensorGemm.scala 132:48]
    dot_3.io.b[12] <= _T_6177 @[TensorGemm.scala 132:22]
    node _T_6178 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_3.io.a[13] <= _T_6178 @[TensorGemm.scala 131:22]
    node _T_6179 = asSInt(io.wgt.data.bits[3][13]) @[TensorGemm.scala 132:48]
    dot_3.io.b[13] <= _T_6179 @[TensorGemm.scala 132:22]
    node _T_6180 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_3.io.a[14] <= _T_6180 @[TensorGemm.scala 131:22]
    node _T_6181 = asSInt(io.wgt.data.bits[3][14]) @[TensorGemm.scala 132:48]
    dot_3.io.b[14] <= _T_6181 @[TensorGemm.scala 132:22]
    node _T_6182 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_3.io.a[15] <= _T_6182 @[TensorGemm.scala 131:22]
    node _T_6183 = asSInt(io.wgt.data.bits[3][15]) @[TensorGemm.scala 132:48]
    dot_3.io.b[15] <= _T_6183 @[TensorGemm.scala 132:22]
    node _T_6184 = asSInt(acc_3.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6185 = add(_T_6184, dot_3.io.y) @[TensorGemm.scala 134:41]
    node _T_6186 = tail(_T_6185, 1) @[TensorGemm.scala 134:41]
    node _T_6187 = asSInt(_T_6186) @[TensorGemm.scala 134:41]
    add_3 <= _T_6187 @[TensorGemm.scala 134:12]
    node _T_6189 = asUInt(add_3) @[TensorGemm.scala 135:59]
    node _T_6190 = mux(io.reset, UInt<1>("h00"), _T_6189) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][3] <= _T_6190 @[TensorGemm.scala 135:30]
    node _T_6191 = asUInt(add_3) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][3] <= _T_6191 @[TensorGemm.scala 136:28]
    vld[3] <= acc_3.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6192 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6193 = and(_T_6192, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6194 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6195 = and(_T_6193, _T_6194) @[TensorGemm.scala 128:88]
    acc_4.io.enq.valid <= _T_6195 @[TensorGemm.scala 128:25]
    acc_4.io.enq.bits <= io.acc_i.data.bits[0][4] @[TensorGemm.scala 129:24]
    node _T_6196 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_4.io.a[0] <= _T_6196 @[TensorGemm.scala 131:22]
    node _T_6197 = asSInt(io.wgt.data.bits[4][0]) @[TensorGemm.scala 132:48]
    dot_4.io.b[0] <= _T_6197 @[TensorGemm.scala 132:22]
    node _T_6198 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_4.io.a[1] <= _T_6198 @[TensorGemm.scala 131:22]
    node _T_6199 = asSInt(io.wgt.data.bits[4][1]) @[TensorGemm.scala 132:48]
    dot_4.io.b[1] <= _T_6199 @[TensorGemm.scala 132:22]
    node _T_6200 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_4.io.a[2] <= _T_6200 @[TensorGemm.scala 131:22]
    node _T_6201 = asSInt(io.wgt.data.bits[4][2]) @[TensorGemm.scala 132:48]
    dot_4.io.b[2] <= _T_6201 @[TensorGemm.scala 132:22]
    node _T_6202 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_4.io.a[3] <= _T_6202 @[TensorGemm.scala 131:22]
    node _T_6203 = asSInt(io.wgt.data.bits[4][3]) @[TensorGemm.scala 132:48]
    dot_4.io.b[3] <= _T_6203 @[TensorGemm.scala 132:22]
    node _T_6204 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_4.io.a[4] <= _T_6204 @[TensorGemm.scala 131:22]
    node _T_6205 = asSInt(io.wgt.data.bits[4][4]) @[TensorGemm.scala 132:48]
    dot_4.io.b[4] <= _T_6205 @[TensorGemm.scala 132:22]
    node _T_6206 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_4.io.a[5] <= _T_6206 @[TensorGemm.scala 131:22]
    node _T_6207 = asSInt(io.wgt.data.bits[4][5]) @[TensorGemm.scala 132:48]
    dot_4.io.b[5] <= _T_6207 @[TensorGemm.scala 132:22]
    node _T_6208 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_4.io.a[6] <= _T_6208 @[TensorGemm.scala 131:22]
    node _T_6209 = asSInt(io.wgt.data.bits[4][6]) @[TensorGemm.scala 132:48]
    dot_4.io.b[6] <= _T_6209 @[TensorGemm.scala 132:22]
    node _T_6210 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_4.io.a[7] <= _T_6210 @[TensorGemm.scala 131:22]
    node _T_6211 = asSInt(io.wgt.data.bits[4][7]) @[TensorGemm.scala 132:48]
    dot_4.io.b[7] <= _T_6211 @[TensorGemm.scala 132:22]
    node _T_6212 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_4.io.a[8] <= _T_6212 @[TensorGemm.scala 131:22]
    node _T_6213 = asSInt(io.wgt.data.bits[4][8]) @[TensorGemm.scala 132:48]
    dot_4.io.b[8] <= _T_6213 @[TensorGemm.scala 132:22]
    node _T_6214 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_4.io.a[9] <= _T_6214 @[TensorGemm.scala 131:22]
    node _T_6215 = asSInt(io.wgt.data.bits[4][9]) @[TensorGemm.scala 132:48]
    dot_4.io.b[9] <= _T_6215 @[TensorGemm.scala 132:22]
    node _T_6216 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_4.io.a[10] <= _T_6216 @[TensorGemm.scala 131:22]
    node _T_6217 = asSInt(io.wgt.data.bits[4][10]) @[TensorGemm.scala 132:48]
    dot_4.io.b[10] <= _T_6217 @[TensorGemm.scala 132:22]
    node _T_6218 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_4.io.a[11] <= _T_6218 @[TensorGemm.scala 131:22]
    node _T_6219 = asSInt(io.wgt.data.bits[4][11]) @[TensorGemm.scala 132:48]
    dot_4.io.b[11] <= _T_6219 @[TensorGemm.scala 132:22]
    node _T_6220 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_4.io.a[12] <= _T_6220 @[TensorGemm.scala 131:22]
    node _T_6221 = asSInt(io.wgt.data.bits[4][12]) @[TensorGemm.scala 132:48]
    dot_4.io.b[12] <= _T_6221 @[TensorGemm.scala 132:22]
    node _T_6222 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_4.io.a[13] <= _T_6222 @[TensorGemm.scala 131:22]
    node _T_6223 = asSInt(io.wgt.data.bits[4][13]) @[TensorGemm.scala 132:48]
    dot_4.io.b[13] <= _T_6223 @[TensorGemm.scala 132:22]
    node _T_6224 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_4.io.a[14] <= _T_6224 @[TensorGemm.scala 131:22]
    node _T_6225 = asSInt(io.wgt.data.bits[4][14]) @[TensorGemm.scala 132:48]
    dot_4.io.b[14] <= _T_6225 @[TensorGemm.scala 132:22]
    node _T_6226 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_4.io.a[15] <= _T_6226 @[TensorGemm.scala 131:22]
    node _T_6227 = asSInt(io.wgt.data.bits[4][15]) @[TensorGemm.scala 132:48]
    dot_4.io.b[15] <= _T_6227 @[TensorGemm.scala 132:22]
    node _T_6228 = asSInt(acc_4.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6229 = add(_T_6228, dot_4.io.y) @[TensorGemm.scala 134:41]
    node _T_6230 = tail(_T_6229, 1) @[TensorGemm.scala 134:41]
    node _T_6231 = asSInt(_T_6230) @[TensorGemm.scala 134:41]
    add_4 <= _T_6231 @[TensorGemm.scala 134:12]
    node _T_6233 = asUInt(add_4) @[TensorGemm.scala 135:59]
    node _T_6234 = mux(io.reset, UInt<1>("h00"), _T_6233) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][4] <= _T_6234 @[TensorGemm.scala 135:30]
    node _T_6235 = asUInt(add_4) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][4] <= _T_6235 @[TensorGemm.scala 136:28]
    vld[4] <= acc_4.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6236 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6237 = and(_T_6236, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6238 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6239 = and(_T_6237, _T_6238) @[TensorGemm.scala 128:88]
    acc_5.io.enq.valid <= _T_6239 @[TensorGemm.scala 128:25]
    acc_5.io.enq.bits <= io.acc_i.data.bits[0][5] @[TensorGemm.scala 129:24]
    node _T_6240 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_5.io.a[0] <= _T_6240 @[TensorGemm.scala 131:22]
    node _T_6241 = asSInt(io.wgt.data.bits[5][0]) @[TensorGemm.scala 132:48]
    dot_5.io.b[0] <= _T_6241 @[TensorGemm.scala 132:22]
    node _T_6242 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_5.io.a[1] <= _T_6242 @[TensorGemm.scala 131:22]
    node _T_6243 = asSInt(io.wgt.data.bits[5][1]) @[TensorGemm.scala 132:48]
    dot_5.io.b[1] <= _T_6243 @[TensorGemm.scala 132:22]
    node _T_6244 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_5.io.a[2] <= _T_6244 @[TensorGemm.scala 131:22]
    node _T_6245 = asSInt(io.wgt.data.bits[5][2]) @[TensorGemm.scala 132:48]
    dot_5.io.b[2] <= _T_6245 @[TensorGemm.scala 132:22]
    node _T_6246 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_5.io.a[3] <= _T_6246 @[TensorGemm.scala 131:22]
    node _T_6247 = asSInt(io.wgt.data.bits[5][3]) @[TensorGemm.scala 132:48]
    dot_5.io.b[3] <= _T_6247 @[TensorGemm.scala 132:22]
    node _T_6248 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_5.io.a[4] <= _T_6248 @[TensorGemm.scala 131:22]
    node _T_6249 = asSInt(io.wgt.data.bits[5][4]) @[TensorGemm.scala 132:48]
    dot_5.io.b[4] <= _T_6249 @[TensorGemm.scala 132:22]
    node _T_6250 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_5.io.a[5] <= _T_6250 @[TensorGemm.scala 131:22]
    node _T_6251 = asSInt(io.wgt.data.bits[5][5]) @[TensorGemm.scala 132:48]
    dot_5.io.b[5] <= _T_6251 @[TensorGemm.scala 132:22]
    node _T_6252 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_5.io.a[6] <= _T_6252 @[TensorGemm.scala 131:22]
    node _T_6253 = asSInt(io.wgt.data.bits[5][6]) @[TensorGemm.scala 132:48]
    dot_5.io.b[6] <= _T_6253 @[TensorGemm.scala 132:22]
    node _T_6254 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_5.io.a[7] <= _T_6254 @[TensorGemm.scala 131:22]
    node _T_6255 = asSInt(io.wgt.data.bits[5][7]) @[TensorGemm.scala 132:48]
    dot_5.io.b[7] <= _T_6255 @[TensorGemm.scala 132:22]
    node _T_6256 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_5.io.a[8] <= _T_6256 @[TensorGemm.scala 131:22]
    node _T_6257 = asSInt(io.wgt.data.bits[5][8]) @[TensorGemm.scala 132:48]
    dot_5.io.b[8] <= _T_6257 @[TensorGemm.scala 132:22]
    node _T_6258 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_5.io.a[9] <= _T_6258 @[TensorGemm.scala 131:22]
    node _T_6259 = asSInt(io.wgt.data.bits[5][9]) @[TensorGemm.scala 132:48]
    dot_5.io.b[9] <= _T_6259 @[TensorGemm.scala 132:22]
    node _T_6260 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_5.io.a[10] <= _T_6260 @[TensorGemm.scala 131:22]
    node _T_6261 = asSInt(io.wgt.data.bits[5][10]) @[TensorGemm.scala 132:48]
    dot_5.io.b[10] <= _T_6261 @[TensorGemm.scala 132:22]
    node _T_6262 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_5.io.a[11] <= _T_6262 @[TensorGemm.scala 131:22]
    node _T_6263 = asSInt(io.wgt.data.bits[5][11]) @[TensorGemm.scala 132:48]
    dot_5.io.b[11] <= _T_6263 @[TensorGemm.scala 132:22]
    node _T_6264 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_5.io.a[12] <= _T_6264 @[TensorGemm.scala 131:22]
    node _T_6265 = asSInt(io.wgt.data.bits[5][12]) @[TensorGemm.scala 132:48]
    dot_5.io.b[12] <= _T_6265 @[TensorGemm.scala 132:22]
    node _T_6266 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_5.io.a[13] <= _T_6266 @[TensorGemm.scala 131:22]
    node _T_6267 = asSInt(io.wgt.data.bits[5][13]) @[TensorGemm.scala 132:48]
    dot_5.io.b[13] <= _T_6267 @[TensorGemm.scala 132:22]
    node _T_6268 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_5.io.a[14] <= _T_6268 @[TensorGemm.scala 131:22]
    node _T_6269 = asSInt(io.wgt.data.bits[5][14]) @[TensorGemm.scala 132:48]
    dot_5.io.b[14] <= _T_6269 @[TensorGemm.scala 132:22]
    node _T_6270 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_5.io.a[15] <= _T_6270 @[TensorGemm.scala 131:22]
    node _T_6271 = asSInt(io.wgt.data.bits[5][15]) @[TensorGemm.scala 132:48]
    dot_5.io.b[15] <= _T_6271 @[TensorGemm.scala 132:22]
    node _T_6272 = asSInt(acc_5.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6273 = add(_T_6272, dot_5.io.y) @[TensorGemm.scala 134:41]
    node _T_6274 = tail(_T_6273, 1) @[TensorGemm.scala 134:41]
    node _T_6275 = asSInt(_T_6274) @[TensorGemm.scala 134:41]
    add_5 <= _T_6275 @[TensorGemm.scala 134:12]
    node _T_6277 = asUInt(add_5) @[TensorGemm.scala 135:59]
    node _T_6278 = mux(io.reset, UInt<1>("h00"), _T_6277) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][5] <= _T_6278 @[TensorGemm.scala 135:30]
    node _T_6279 = asUInt(add_5) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][5] <= _T_6279 @[TensorGemm.scala 136:28]
    vld[5] <= acc_5.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6280 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6281 = and(_T_6280, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6282 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6283 = and(_T_6281, _T_6282) @[TensorGemm.scala 128:88]
    acc_6.io.enq.valid <= _T_6283 @[TensorGemm.scala 128:25]
    acc_6.io.enq.bits <= io.acc_i.data.bits[0][6] @[TensorGemm.scala 129:24]
    node _T_6284 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_6.io.a[0] <= _T_6284 @[TensorGemm.scala 131:22]
    node _T_6285 = asSInt(io.wgt.data.bits[6][0]) @[TensorGemm.scala 132:48]
    dot_6.io.b[0] <= _T_6285 @[TensorGemm.scala 132:22]
    node _T_6286 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_6.io.a[1] <= _T_6286 @[TensorGemm.scala 131:22]
    node _T_6287 = asSInt(io.wgt.data.bits[6][1]) @[TensorGemm.scala 132:48]
    dot_6.io.b[1] <= _T_6287 @[TensorGemm.scala 132:22]
    node _T_6288 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_6.io.a[2] <= _T_6288 @[TensorGemm.scala 131:22]
    node _T_6289 = asSInt(io.wgt.data.bits[6][2]) @[TensorGemm.scala 132:48]
    dot_6.io.b[2] <= _T_6289 @[TensorGemm.scala 132:22]
    node _T_6290 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_6.io.a[3] <= _T_6290 @[TensorGemm.scala 131:22]
    node _T_6291 = asSInt(io.wgt.data.bits[6][3]) @[TensorGemm.scala 132:48]
    dot_6.io.b[3] <= _T_6291 @[TensorGemm.scala 132:22]
    node _T_6292 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_6.io.a[4] <= _T_6292 @[TensorGemm.scala 131:22]
    node _T_6293 = asSInt(io.wgt.data.bits[6][4]) @[TensorGemm.scala 132:48]
    dot_6.io.b[4] <= _T_6293 @[TensorGemm.scala 132:22]
    node _T_6294 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_6.io.a[5] <= _T_6294 @[TensorGemm.scala 131:22]
    node _T_6295 = asSInt(io.wgt.data.bits[6][5]) @[TensorGemm.scala 132:48]
    dot_6.io.b[5] <= _T_6295 @[TensorGemm.scala 132:22]
    node _T_6296 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_6.io.a[6] <= _T_6296 @[TensorGemm.scala 131:22]
    node _T_6297 = asSInt(io.wgt.data.bits[6][6]) @[TensorGemm.scala 132:48]
    dot_6.io.b[6] <= _T_6297 @[TensorGemm.scala 132:22]
    node _T_6298 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_6.io.a[7] <= _T_6298 @[TensorGemm.scala 131:22]
    node _T_6299 = asSInt(io.wgt.data.bits[6][7]) @[TensorGemm.scala 132:48]
    dot_6.io.b[7] <= _T_6299 @[TensorGemm.scala 132:22]
    node _T_6300 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_6.io.a[8] <= _T_6300 @[TensorGemm.scala 131:22]
    node _T_6301 = asSInt(io.wgt.data.bits[6][8]) @[TensorGemm.scala 132:48]
    dot_6.io.b[8] <= _T_6301 @[TensorGemm.scala 132:22]
    node _T_6302 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_6.io.a[9] <= _T_6302 @[TensorGemm.scala 131:22]
    node _T_6303 = asSInt(io.wgt.data.bits[6][9]) @[TensorGemm.scala 132:48]
    dot_6.io.b[9] <= _T_6303 @[TensorGemm.scala 132:22]
    node _T_6304 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_6.io.a[10] <= _T_6304 @[TensorGemm.scala 131:22]
    node _T_6305 = asSInt(io.wgt.data.bits[6][10]) @[TensorGemm.scala 132:48]
    dot_6.io.b[10] <= _T_6305 @[TensorGemm.scala 132:22]
    node _T_6306 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_6.io.a[11] <= _T_6306 @[TensorGemm.scala 131:22]
    node _T_6307 = asSInt(io.wgt.data.bits[6][11]) @[TensorGemm.scala 132:48]
    dot_6.io.b[11] <= _T_6307 @[TensorGemm.scala 132:22]
    node _T_6308 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_6.io.a[12] <= _T_6308 @[TensorGemm.scala 131:22]
    node _T_6309 = asSInt(io.wgt.data.bits[6][12]) @[TensorGemm.scala 132:48]
    dot_6.io.b[12] <= _T_6309 @[TensorGemm.scala 132:22]
    node _T_6310 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_6.io.a[13] <= _T_6310 @[TensorGemm.scala 131:22]
    node _T_6311 = asSInt(io.wgt.data.bits[6][13]) @[TensorGemm.scala 132:48]
    dot_6.io.b[13] <= _T_6311 @[TensorGemm.scala 132:22]
    node _T_6312 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_6.io.a[14] <= _T_6312 @[TensorGemm.scala 131:22]
    node _T_6313 = asSInt(io.wgt.data.bits[6][14]) @[TensorGemm.scala 132:48]
    dot_6.io.b[14] <= _T_6313 @[TensorGemm.scala 132:22]
    node _T_6314 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_6.io.a[15] <= _T_6314 @[TensorGemm.scala 131:22]
    node _T_6315 = asSInt(io.wgt.data.bits[6][15]) @[TensorGemm.scala 132:48]
    dot_6.io.b[15] <= _T_6315 @[TensorGemm.scala 132:22]
    node _T_6316 = asSInt(acc_6.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6317 = add(_T_6316, dot_6.io.y) @[TensorGemm.scala 134:41]
    node _T_6318 = tail(_T_6317, 1) @[TensorGemm.scala 134:41]
    node _T_6319 = asSInt(_T_6318) @[TensorGemm.scala 134:41]
    add_6 <= _T_6319 @[TensorGemm.scala 134:12]
    node _T_6321 = asUInt(add_6) @[TensorGemm.scala 135:59]
    node _T_6322 = mux(io.reset, UInt<1>("h00"), _T_6321) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][6] <= _T_6322 @[TensorGemm.scala 135:30]
    node _T_6323 = asUInt(add_6) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][6] <= _T_6323 @[TensorGemm.scala 136:28]
    vld[6] <= acc_6.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6324 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6325 = and(_T_6324, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6326 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6327 = and(_T_6325, _T_6326) @[TensorGemm.scala 128:88]
    acc_7.io.enq.valid <= _T_6327 @[TensorGemm.scala 128:25]
    acc_7.io.enq.bits <= io.acc_i.data.bits[0][7] @[TensorGemm.scala 129:24]
    node _T_6328 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_7.io.a[0] <= _T_6328 @[TensorGemm.scala 131:22]
    node _T_6329 = asSInt(io.wgt.data.bits[7][0]) @[TensorGemm.scala 132:48]
    dot_7.io.b[0] <= _T_6329 @[TensorGemm.scala 132:22]
    node _T_6330 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_7.io.a[1] <= _T_6330 @[TensorGemm.scala 131:22]
    node _T_6331 = asSInt(io.wgt.data.bits[7][1]) @[TensorGemm.scala 132:48]
    dot_7.io.b[1] <= _T_6331 @[TensorGemm.scala 132:22]
    node _T_6332 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_7.io.a[2] <= _T_6332 @[TensorGemm.scala 131:22]
    node _T_6333 = asSInt(io.wgt.data.bits[7][2]) @[TensorGemm.scala 132:48]
    dot_7.io.b[2] <= _T_6333 @[TensorGemm.scala 132:22]
    node _T_6334 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_7.io.a[3] <= _T_6334 @[TensorGemm.scala 131:22]
    node _T_6335 = asSInt(io.wgt.data.bits[7][3]) @[TensorGemm.scala 132:48]
    dot_7.io.b[3] <= _T_6335 @[TensorGemm.scala 132:22]
    node _T_6336 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_7.io.a[4] <= _T_6336 @[TensorGemm.scala 131:22]
    node _T_6337 = asSInt(io.wgt.data.bits[7][4]) @[TensorGemm.scala 132:48]
    dot_7.io.b[4] <= _T_6337 @[TensorGemm.scala 132:22]
    node _T_6338 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_7.io.a[5] <= _T_6338 @[TensorGemm.scala 131:22]
    node _T_6339 = asSInt(io.wgt.data.bits[7][5]) @[TensorGemm.scala 132:48]
    dot_7.io.b[5] <= _T_6339 @[TensorGemm.scala 132:22]
    node _T_6340 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_7.io.a[6] <= _T_6340 @[TensorGemm.scala 131:22]
    node _T_6341 = asSInt(io.wgt.data.bits[7][6]) @[TensorGemm.scala 132:48]
    dot_7.io.b[6] <= _T_6341 @[TensorGemm.scala 132:22]
    node _T_6342 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_7.io.a[7] <= _T_6342 @[TensorGemm.scala 131:22]
    node _T_6343 = asSInt(io.wgt.data.bits[7][7]) @[TensorGemm.scala 132:48]
    dot_7.io.b[7] <= _T_6343 @[TensorGemm.scala 132:22]
    node _T_6344 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_7.io.a[8] <= _T_6344 @[TensorGemm.scala 131:22]
    node _T_6345 = asSInt(io.wgt.data.bits[7][8]) @[TensorGemm.scala 132:48]
    dot_7.io.b[8] <= _T_6345 @[TensorGemm.scala 132:22]
    node _T_6346 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_7.io.a[9] <= _T_6346 @[TensorGemm.scala 131:22]
    node _T_6347 = asSInt(io.wgt.data.bits[7][9]) @[TensorGemm.scala 132:48]
    dot_7.io.b[9] <= _T_6347 @[TensorGemm.scala 132:22]
    node _T_6348 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_7.io.a[10] <= _T_6348 @[TensorGemm.scala 131:22]
    node _T_6349 = asSInt(io.wgt.data.bits[7][10]) @[TensorGemm.scala 132:48]
    dot_7.io.b[10] <= _T_6349 @[TensorGemm.scala 132:22]
    node _T_6350 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_7.io.a[11] <= _T_6350 @[TensorGemm.scala 131:22]
    node _T_6351 = asSInt(io.wgt.data.bits[7][11]) @[TensorGemm.scala 132:48]
    dot_7.io.b[11] <= _T_6351 @[TensorGemm.scala 132:22]
    node _T_6352 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_7.io.a[12] <= _T_6352 @[TensorGemm.scala 131:22]
    node _T_6353 = asSInt(io.wgt.data.bits[7][12]) @[TensorGemm.scala 132:48]
    dot_7.io.b[12] <= _T_6353 @[TensorGemm.scala 132:22]
    node _T_6354 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_7.io.a[13] <= _T_6354 @[TensorGemm.scala 131:22]
    node _T_6355 = asSInt(io.wgt.data.bits[7][13]) @[TensorGemm.scala 132:48]
    dot_7.io.b[13] <= _T_6355 @[TensorGemm.scala 132:22]
    node _T_6356 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_7.io.a[14] <= _T_6356 @[TensorGemm.scala 131:22]
    node _T_6357 = asSInt(io.wgt.data.bits[7][14]) @[TensorGemm.scala 132:48]
    dot_7.io.b[14] <= _T_6357 @[TensorGemm.scala 132:22]
    node _T_6358 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_7.io.a[15] <= _T_6358 @[TensorGemm.scala 131:22]
    node _T_6359 = asSInt(io.wgt.data.bits[7][15]) @[TensorGemm.scala 132:48]
    dot_7.io.b[15] <= _T_6359 @[TensorGemm.scala 132:22]
    node _T_6360 = asSInt(acc_7.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6361 = add(_T_6360, dot_7.io.y) @[TensorGemm.scala 134:41]
    node _T_6362 = tail(_T_6361, 1) @[TensorGemm.scala 134:41]
    node _T_6363 = asSInt(_T_6362) @[TensorGemm.scala 134:41]
    add_7 <= _T_6363 @[TensorGemm.scala 134:12]
    node _T_6365 = asUInt(add_7) @[TensorGemm.scala 135:59]
    node _T_6366 = mux(io.reset, UInt<1>("h00"), _T_6365) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][7] <= _T_6366 @[TensorGemm.scala 135:30]
    node _T_6367 = asUInt(add_7) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][7] <= _T_6367 @[TensorGemm.scala 136:28]
    vld[7] <= acc_7.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6368 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6369 = and(_T_6368, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6370 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6371 = and(_T_6369, _T_6370) @[TensorGemm.scala 128:88]
    acc_8.io.enq.valid <= _T_6371 @[TensorGemm.scala 128:25]
    acc_8.io.enq.bits <= io.acc_i.data.bits[0][8] @[TensorGemm.scala 129:24]
    node _T_6372 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_8.io.a[0] <= _T_6372 @[TensorGemm.scala 131:22]
    node _T_6373 = asSInt(io.wgt.data.bits[8][0]) @[TensorGemm.scala 132:48]
    dot_8.io.b[0] <= _T_6373 @[TensorGemm.scala 132:22]
    node _T_6374 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_8.io.a[1] <= _T_6374 @[TensorGemm.scala 131:22]
    node _T_6375 = asSInt(io.wgt.data.bits[8][1]) @[TensorGemm.scala 132:48]
    dot_8.io.b[1] <= _T_6375 @[TensorGemm.scala 132:22]
    node _T_6376 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_8.io.a[2] <= _T_6376 @[TensorGemm.scala 131:22]
    node _T_6377 = asSInt(io.wgt.data.bits[8][2]) @[TensorGemm.scala 132:48]
    dot_8.io.b[2] <= _T_6377 @[TensorGemm.scala 132:22]
    node _T_6378 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_8.io.a[3] <= _T_6378 @[TensorGemm.scala 131:22]
    node _T_6379 = asSInt(io.wgt.data.bits[8][3]) @[TensorGemm.scala 132:48]
    dot_8.io.b[3] <= _T_6379 @[TensorGemm.scala 132:22]
    node _T_6380 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_8.io.a[4] <= _T_6380 @[TensorGemm.scala 131:22]
    node _T_6381 = asSInt(io.wgt.data.bits[8][4]) @[TensorGemm.scala 132:48]
    dot_8.io.b[4] <= _T_6381 @[TensorGemm.scala 132:22]
    node _T_6382 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_8.io.a[5] <= _T_6382 @[TensorGemm.scala 131:22]
    node _T_6383 = asSInt(io.wgt.data.bits[8][5]) @[TensorGemm.scala 132:48]
    dot_8.io.b[5] <= _T_6383 @[TensorGemm.scala 132:22]
    node _T_6384 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_8.io.a[6] <= _T_6384 @[TensorGemm.scala 131:22]
    node _T_6385 = asSInt(io.wgt.data.bits[8][6]) @[TensorGemm.scala 132:48]
    dot_8.io.b[6] <= _T_6385 @[TensorGemm.scala 132:22]
    node _T_6386 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_8.io.a[7] <= _T_6386 @[TensorGemm.scala 131:22]
    node _T_6387 = asSInt(io.wgt.data.bits[8][7]) @[TensorGemm.scala 132:48]
    dot_8.io.b[7] <= _T_6387 @[TensorGemm.scala 132:22]
    node _T_6388 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_8.io.a[8] <= _T_6388 @[TensorGemm.scala 131:22]
    node _T_6389 = asSInt(io.wgt.data.bits[8][8]) @[TensorGemm.scala 132:48]
    dot_8.io.b[8] <= _T_6389 @[TensorGemm.scala 132:22]
    node _T_6390 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_8.io.a[9] <= _T_6390 @[TensorGemm.scala 131:22]
    node _T_6391 = asSInt(io.wgt.data.bits[8][9]) @[TensorGemm.scala 132:48]
    dot_8.io.b[9] <= _T_6391 @[TensorGemm.scala 132:22]
    node _T_6392 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_8.io.a[10] <= _T_6392 @[TensorGemm.scala 131:22]
    node _T_6393 = asSInt(io.wgt.data.bits[8][10]) @[TensorGemm.scala 132:48]
    dot_8.io.b[10] <= _T_6393 @[TensorGemm.scala 132:22]
    node _T_6394 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_8.io.a[11] <= _T_6394 @[TensorGemm.scala 131:22]
    node _T_6395 = asSInt(io.wgt.data.bits[8][11]) @[TensorGemm.scala 132:48]
    dot_8.io.b[11] <= _T_6395 @[TensorGemm.scala 132:22]
    node _T_6396 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_8.io.a[12] <= _T_6396 @[TensorGemm.scala 131:22]
    node _T_6397 = asSInt(io.wgt.data.bits[8][12]) @[TensorGemm.scala 132:48]
    dot_8.io.b[12] <= _T_6397 @[TensorGemm.scala 132:22]
    node _T_6398 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_8.io.a[13] <= _T_6398 @[TensorGemm.scala 131:22]
    node _T_6399 = asSInt(io.wgt.data.bits[8][13]) @[TensorGemm.scala 132:48]
    dot_8.io.b[13] <= _T_6399 @[TensorGemm.scala 132:22]
    node _T_6400 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_8.io.a[14] <= _T_6400 @[TensorGemm.scala 131:22]
    node _T_6401 = asSInt(io.wgt.data.bits[8][14]) @[TensorGemm.scala 132:48]
    dot_8.io.b[14] <= _T_6401 @[TensorGemm.scala 132:22]
    node _T_6402 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_8.io.a[15] <= _T_6402 @[TensorGemm.scala 131:22]
    node _T_6403 = asSInt(io.wgt.data.bits[8][15]) @[TensorGemm.scala 132:48]
    dot_8.io.b[15] <= _T_6403 @[TensorGemm.scala 132:22]
    node _T_6404 = asSInt(acc_8.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6405 = add(_T_6404, dot_8.io.y) @[TensorGemm.scala 134:41]
    node _T_6406 = tail(_T_6405, 1) @[TensorGemm.scala 134:41]
    node _T_6407 = asSInt(_T_6406) @[TensorGemm.scala 134:41]
    add_8 <= _T_6407 @[TensorGemm.scala 134:12]
    node _T_6409 = asUInt(add_8) @[TensorGemm.scala 135:59]
    node _T_6410 = mux(io.reset, UInt<1>("h00"), _T_6409) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][8] <= _T_6410 @[TensorGemm.scala 135:30]
    node _T_6411 = asUInt(add_8) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][8] <= _T_6411 @[TensorGemm.scala 136:28]
    vld[8] <= acc_8.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6412 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6413 = and(_T_6412, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6414 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6415 = and(_T_6413, _T_6414) @[TensorGemm.scala 128:88]
    acc_9.io.enq.valid <= _T_6415 @[TensorGemm.scala 128:25]
    acc_9.io.enq.bits <= io.acc_i.data.bits[0][9] @[TensorGemm.scala 129:24]
    node _T_6416 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_9.io.a[0] <= _T_6416 @[TensorGemm.scala 131:22]
    node _T_6417 = asSInt(io.wgt.data.bits[9][0]) @[TensorGemm.scala 132:48]
    dot_9.io.b[0] <= _T_6417 @[TensorGemm.scala 132:22]
    node _T_6418 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_9.io.a[1] <= _T_6418 @[TensorGemm.scala 131:22]
    node _T_6419 = asSInt(io.wgt.data.bits[9][1]) @[TensorGemm.scala 132:48]
    dot_9.io.b[1] <= _T_6419 @[TensorGemm.scala 132:22]
    node _T_6420 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_9.io.a[2] <= _T_6420 @[TensorGemm.scala 131:22]
    node _T_6421 = asSInt(io.wgt.data.bits[9][2]) @[TensorGemm.scala 132:48]
    dot_9.io.b[2] <= _T_6421 @[TensorGemm.scala 132:22]
    node _T_6422 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_9.io.a[3] <= _T_6422 @[TensorGemm.scala 131:22]
    node _T_6423 = asSInt(io.wgt.data.bits[9][3]) @[TensorGemm.scala 132:48]
    dot_9.io.b[3] <= _T_6423 @[TensorGemm.scala 132:22]
    node _T_6424 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_9.io.a[4] <= _T_6424 @[TensorGemm.scala 131:22]
    node _T_6425 = asSInt(io.wgt.data.bits[9][4]) @[TensorGemm.scala 132:48]
    dot_9.io.b[4] <= _T_6425 @[TensorGemm.scala 132:22]
    node _T_6426 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_9.io.a[5] <= _T_6426 @[TensorGemm.scala 131:22]
    node _T_6427 = asSInt(io.wgt.data.bits[9][5]) @[TensorGemm.scala 132:48]
    dot_9.io.b[5] <= _T_6427 @[TensorGemm.scala 132:22]
    node _T_6428 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_9.io.a[6] <= _T_6428 @[TensorGemm.scala 131:22]
    node _T_6429 = asSInt(io.wgt.data.bits[9][6]) @[TensorGemm.scala 132:48]
    dot_9.io.b[6] <= _T_6429 @[TensorGemm.scala 132:22]
    node _T_6430 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_9.io.a[7] <= _T_6430 @[TensorGemm.scala 131:22]
    node _T_6431 = asSInt(io.wgt.data.bits[9][7]) @[TensorGemm.scala 132:48]
    dot_9.io.b[7] <= _T_6431 @[TensorGemm.scala 132:22]
    node _T_6432 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_9.io.a[8] <= _T_6432 @[TensorGemm.scala 131:22]
    node _T_6433 = asSInt(io.wgt.data.bits[9][8]) @[TensorGemm.scala 132:48]
    dot_9.io.b[8] <= _T_6433 @[TensorGemm.scala 132:22]
    node _T_6434 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_9.io.a[9] <= _T_6434 @[TensorGemm.scala 131:22]
    node _T_6435 = asSInt(io.wgt.data.bits[9][9]) @[TensorGemm.scala 132:48]
    dot_9.io.b[9] <= _T_6435 @[TensorGemm.scala 132:22]
    node _T_6436 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_9.io.a[10] <= _T_6436 @[TensorGemm.scala 131:22]
    node _T_6437 = asSInt(io.wgt.data.bits[9][10]) @[TensorGemm.scala 132:48]
    dot_9.io.b[10] <= _T_6437 @[TensorGemm.scala 132:22]
    node _T_6438 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_9.io.a[11] <= _T_6438 @[TensorGemm.scala 131:22]
    node _T_6439 = asSInt(io.wgt.data.bits[9][11]) @[TensorGemm.scala 132:48]
    dot_9.io.b[11] <= _T_6439 @[TensorGemm.scala 132:22]
    node _T_6440 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_9.io.a[12] <= _T_6440 @[TensorGemm.scala 131:22]
    node _T_6441 = asSInt(io.wgt.data.bits[9][12]) @[TensorGemm.scala 132:48]
    dot_9.io.b[12] <= _T_6441 @[TensorGemm.scala 132:22]
    node _T_6442 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_9.io.a[13] <= _T_6442 @[TensorGemm.scala 131:22]
    node _T_6443 = asSInt(io.wgt.data.bits[9][13]) @[TensorGemm.scala 132:48]
    dot_9.io.b[13] <= _T_6443 @[TensorGemm.scala 132:22]
    node _T_6444 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_9.io.a[14] <= _T_6444 @[TensorGemm.scala 131:22]
    node _T_6445 = asSInt(io.wgt.data.bits[9][14]) @[TensorGemm.scala 132:48]
    dot_9.io.b[14] <= _T_6445 @[TensorGemm.scala 132:22]
    node _T_6446 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_9.io.a[15] <= _T_6446 @[TensorGemm.scala 131:22]
    node _T_6447 = asSInt(io.wgt.data.bits[9][15]) @[TensorGemm.scala 132:48]
    dot_9.io.b[15] <= _T_6447 @[TensorGemm.scala 132:22]
    node _T_6448 = asSInt(acc_9.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6449 = add(_T_6448, dot_9.io.y) @[TensorGemm.scala 134:41]
    node _T_6450 = tail(_T_6449, 1) @[TensorGemm.scala 134:41]
    node _T_6451 = asSInt(_T_6450) @[TensorGemm.scala 134:41]
    add_9 <= _T_6451 @[TensorGemm.scala 134:12]
    node _T_6453 = asUInt(add_9) @[TensorGemm.scala 135:59]
    node _T_6454 = mux(io.reset, UInt<1>("h00"), _T_6453) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][9] <= _T_6454 @[TensorGemm.scala 135:30]
    node _T_6455 = asUInt(add_9) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][9] <= _T_6455 @[TensorGemm.scala 136:28]
    vld[9] <= acc_9.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6456 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6457 = and(_T_6456, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6458 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6459 = and(_T_6457, _T_6458) @[TensorGemm.scala 128:88]
    acc_10.io.enq.valid <= _T_6459 @[TensorGemm.scala 128:25]
    acc_10.io.enq.bits <= io.acc_i.data.bits[0][10] @[TensorGemm.scala 129:24]
    node _T_6460 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_10.io.a[0] <= _T_6460 @[TensorGemm.scala 131:22]
    node _T_6461 = asSInt(io.wgt.data.bits[10][0]) @[TensorGemm.scala 132:48]
    dot_10.io.b[0] <= _T_6461 @[TensorGemm.scala 132:22]
    node _T_6462 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_10.io.a[1] <= _T_6462 @[TensorGemm.scala 131:22]
    node _T_6463 = asSInt(io.wgt.data.bits[10][1]) @[TensorGemm.scala 132:48]
    dot_10.io.b[1] <= _T_6463 @[TensorGemm.scala 132:22]
    node _T_6464 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_10.io.a[2] <= _T_6464 @[TensorGemm.scala 131:22]
    node _T_6465 = asSInt(io.wgt.data.bits[10][2]) @[TensorGemm.scala 132:48]
    dot_10.io.b[2] <= _T_6465 @[TensorGemm.scala 132:22]
    node _T_6466 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_10.io.a[3] <= _T_6466 @[TensorGemm.scala 131:22]
    node _T_6467 = asSInt(io.wgt.data.bits[10][3]) @[TensorGemm.scala 132:48]
    dot_10.io.b[3] <= _T_6467 @[TensorGemm.scala 132:22]
    node _T_6468 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_10.io.a[4] <= _T_6468 @[TensorGemm.scala 131:22]
    node _T_6469 = asSInt(io.wgt.data.bits[10][4]) @[TensorGemm.scala 132:48]
    dot_10.io.b[4] <= _T_6469 @[TensorGemm.scala 132:22]
    node _T_6470 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_10.io.a[5] <= _T_6470 @[TensorGemm.scala 131:22]
    node _T_6471 = asSInt(io.wgt.data.bits[10][5]) @[TensorGemm.scala 132:48]
    dot_10.io.b[5] <= _T_6471 @[TensorGemm.scala 132:22]
    node _T_6472 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_10.io.a[6] <= _T_6472 @[TensorGemm.scala 131:22]
    node _T_6473 = asSInt(io.wgt.data.bits[10][6]) @[TensorGemm.scala 132:48]
    dot_10.io.b[6] <= _T_6473 @[TensorGemm.scala 132:22]
    node _T_6474 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_10.io.a[7] <= _T_6474 @[TensorGemm.scala 131:22]
    node _T_6475 = asSInt(io.wgt.data.bits[10][7]) @[TensorGemm.scala 132:48]
    dot_10.io.b[7] <= _T_6475 @[TensorGemm.scala 132:22]
    node _T_6476 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_10.io.a[8] <= _T_6476 @[TensorGemm.scala 131:22]
    node _T_6477 = asSInt(io.wgt.data.bits[10][8]) @[TensorGemm.scala 132:48]
    dot_10.io.b[8] <= _T_6477 @[TensorGemm.scala 132:22]
    node _T_6478 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_10.io.a[9] <= _T_6478 @[TensorGemm.scala 131:22]
    node _T_6479 = asSInt(io.wgt.data.bits[10][9]) @[TensorGemm.scala 132:48]
    dot_10.io.b[9] <= _T_6479 @[TensorGemm.scala 132:22]
    node _T_6480 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_10.io.a[10] <= _T_6480 @[TensorGemm.scala 131:22]
    node _T_6481 = asSInt(io.wgt.data.bits[10][10]) @[TensorGemm.scala 132:48]
    dot_10.io.b[10] <= _T_6481 @[TensorGemm.scala 132:22]
    node _T_6482 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_10.io.a[11] <= _T_6482 @[TensorGemm.scala 131:22]
    node _T_6483 = asSInt(io.wgt.data.bits[10][11]) @[TensorGemm.scala 132:48]
    dot_10.io.b[11] <= _T_6483 @[TensorGemm.scala 132:22]
    node _T_6484 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_10.io.a[12] <= _T_6484 @[TensorGemm.scala 131:22]
    node _T_6485 = asSInt(io.wgt.data.bits[10][12]) @[TensorGemm.scala 132:48]
    dot_10.io.b[12] <= _T_6485 @[TensorGemm.scala 132:22]
    node _T_6486 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_10.io.a[13] <= _T_6486 @[TensorGemm.scala 131:22]
    node _T_6487 = asSInt(io.wgt.data.bits[10][13]) @[TensorGemm.scala 132:48]
    dot_10.io.b[13] <= _T_6487 @[TensorGemm.scala 132:22]
    node _T_6488 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_10.io.a[14] <= _T_6488 @[TensorGemm.scala 131:22]
    node _T_6489 = asSInt(io.wgt.data.bits[10][14]) @[TensorGemm.scala 132:48]
    dot_10.io.b[14] <= _T_6489 @[TensorGemm.scala 132:22]
    node _T_6490 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_10.io.a[15] <= _T_6490 @[TensorGemm.scala 131:22]
    node _T_6491 = asSInt(io.wgt.data.bits[10][15]) @[TensorGemm.scala 132:48]
    dot_10.io.b[15] <= _T_6491 @[TensorGemm.scala 132:22]
    node _T_6492 = asSInt(acc_10.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6493 = add(_T_6492, dot_10.io.y) @[TensorGemm.scala 134:41]
    node _T_6494 = tail(_T_6493, 1) @[TensorGemm.scala 134:41]
    node _T_6495 = asSInt(_T_6494) @[TensorGemm.scala 134:41]
    add_10 <= _T_6495 @[TensorGemm.scala 134:12]
    node _T_6497 = asUInt(add_10) @[TensorGemm.scala 135:59]
    node _T_6498 = mux(io.reset, UInt<1>("h00"), _T_6497) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][10] <= _T_6498 @[TensorGemm.scala 135:30]
    node _T_6499 = asUInt(add_10) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][10] <= _T_6499 @[TensorGemm.scala 136:28]
    vld[10] <= acc_10.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6500 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6501 = and(_T_6500, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6502 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6503 = and(_T_6501, _T_6502) @[TensorGemm.scala 128:88]
    acc_11.io.enq.valid <= _T_6503 @[TensorGemm.scala 128:25]
    acc_11.io.enq.bits <= io.acc_i.data.bits[0][11] @[TensorGemm.scala 129:24]
    node _T_6504 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_11.io.a[0] <= _T_6504 @[TensorGemm.scala 131:22]
    node _T_6505 = asSInt(io.wgt.data.bits[11][0]) @[TensorGemm.scala 132:48]
    dot_11.io.b[0] <= _T_6505 @[TensorGemm.scala 132:22]
    node _T_6506 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_11.io.a[1] <= _T_6506 @[TensorGemm.scala 131:22]
    node _T_6507 = asSInt(io.wgt.data.bits[11][1]) @[TensorGemm.scala 132:48]
    dot_11.io.b[1] <= _T_6507 @[TensorGemm.scala 132:22]
    node _T_6508 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_11.io.a[2] <= _T_6508 @[TensorGemm.scala 131:22]
    node _T_6509 = asSInt(io.wgt.data.bits[11][2]) @[TensorGemm.scala 132:48]
    dot_11.io.b[2] <= _T_6509 @[TensorGemm.scala 132:22]
    node _T_6510 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_11.io.a[3] <= _T_6510 @[TensorGemm.scala 131:22]
    node _T_6511 = asSInt(io.wgt.data.bits[11][3]) @[TensorGemm.scala 132:48]
    dot_11.io.b[3] <= _T_6511 @[TensorGemm.scala 132:22]
    node _T_6512 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_11.io.a[4] <= _T_6512 @[TensorGemm.scala 131:22]
    node _T_6513 = asSInt(io.wgt.data.bits[11][4]) @[TensorGemm.scala 132:48]
    dot_11.io.b[4] <= _T_6513 @[TensorGemm.scala 132:22]
    node _T_6514 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_11.io.a[5] <= _T_6514 @[TensorGemm.scala 131:22]
    node _T_6515 = asSInt(io.wgt.data.bits[11][5]) @[TensorGemm.scala 132:48]
    dot_11.io.b[5] <= _T_6515 @[TensorGemm.scala 132:22]
    node _T_6516 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_11.io.a[6] <= _T_6516 @[TensorGemm.scala 131:22]
    node _T_6517 = asSInt(io.wgt.data.bits[11][6]) @[TensorGemm.scala 132:48]
    dot_11.io.b[6] <= _T_6517 @[TensorGemm.scala 132:22]
    node _T_6518 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_11.io.a[7] <= _T_6518 @[TensorGemm.scala 131:22]
    node _T_6519 = asSInt(io.wgt.data.bits[11][7]) @[TensorGemm.scala 132:48]
    dot_11.io.b[7] <= _T_6519 @[TensorGemm.scala 132:22]
    node _T_6520 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_11.io.a[8] <= _T_6520 @[TensorGemm.scala 131:22]
    node _T_6521 = asSInt(io.wgt.data.bits[11][8]) @[TensorGemm.scala 132:48]
    dot_11.io.b[8] <= _T_6521 @[TensorGemm.scala 132:22]
    node _T_6522 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_11.io.a[9] <= _T_6522 @[TensorGemm.scala 131:22]
    node _T_6523 = asSInt(io.wgt.data.bits[11][9]) @[TensorGemm.scala 132:48]
    dot_11.io.b[9] <= _T_6523 @[TensorGemm.scala 132:22]
    node _T_6524 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_11.io.a[10] <= _T_6524 @[TensorGemm.scala 131:22]
    node _T_6525 = asSInt(io.wgt.data.bits[11][10]) @[TensorGemm.scala 132:48]
    dot_11.io.b[10] <= _T_6525 @[TensorGemm.scala 132:22]
    node _T_6526 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_11.io.a[11] <= _T_6526 @[TensorGemm.scala 131:22]
    node _T_6527 = asSInt(io.wgt.data.bits[11][11]) @[TensorGemm.scala 132:48]
    dot_11.io.b[11] <= _T_6527 @[TensorGemm.scala 132:22]
    node _T_6528 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_11.io.a[12] <= _T_6528 @[TensorGemm.scala 131:22]
    node _T_6529 = asSInt(io.wgt.data.bits[11][12]) @[TensorGemm.scala 132:48]
    dot_11.io.b[12] <= _T_6529 @[TensorGemm.scala 132:22]
    node _T_6530 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_11.io.a[13] <= _T_6530 @[TensorGemm.scala 131:22]
    node _T_6531 = asSInt(io.wgt.data.bits[11][13]) @[TensorGemm.scala 132:48]
    dot_11.io.b[13] <= _T_6531 @[TensorGemm.scala 132:22]
    node _T_6532 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_11.io.a[14] <= _T_6532 @[TensorGemm.scala 131:22]
    node _T_6533 = asSInt(io.wgt.data.bits[11][14]) @[TensorGemm.scala 132:48]
    dot_11.io.b[14] <= _T_6533 @[TensorGemm.scala 132:22]
    node _T_6534 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_11.io.a[15] <= _T_6534 @[TensorGemm.scala 131:22]
    node _T_6535 = asSInt(io.wgt.data.bits[11][15]) @[TensorGemm.scala 132:48]
    dot_11.io.b[15] <= _T_6535 @[TensorGemm.scala 132:22]
    node _T_6536 = asSInt(acc_11.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6537 = add(_T_6536, dot_11.io.y) @[TensorGemm.scala 134:41]
    node _T_6538 = tail(_T_6537, 1) @[TensorGemm.scala 134:41]
    node _T_6539 = asSInt(_T_6538) @[TensorGemm.scala 134:41]
    add_11 <= _T_6539 @[TensorGemm.scala 134:12]
    node _T_6541 = asUInt(add_11) @[TensorGemm.scala 135:59]
    node _T_6542 = mux(io.reset, UInt<1>("h00"), _T_6541) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][11] <= _T_6542 @[TensorGemm.scala 135:30]
    node _T_6543 = asUInt(add_11) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][11] <= _T_6543 @[TensorGemm.scala 136:28]
    vld[11] <= acc_11.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6544 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6545 = and(_T_6544, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6546 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6547 = and(_T_6545, _T_6546) @[TensorGemm.scala 128:88]
    acc_12.io.enq.valid <= _T_6547 @[TensorGemm.scala 128:25]
    acc_12.io.enq.bits <= io.acc_i.data.bits[0][12] @[TensorGemm.scala 129:24]
    node _T_6548 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_12.io.a[0] <= _T_6548 @[TensorGemm.scala 131:22]
    node _T_6549 = asSInt(io.wgt.data.bits[12][0]) @[TensorGemm.scala 132:48]
    dot_12.io.b[0] <= _T_6549 @[TensorGemm.scala 132:22]
    node _T_6550 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_12.io.a[1] <= _T_6550 @[TensorGemm.scala 131:22]
    node _T_6551 = asSInt(io.wgt.data.bits[12][1]) @[TensorGemm.scala 132:48]
    dot_12.io.b[1] <= _T_6551 @[TensorGemm.scala 132:22]
    node _T_6552 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_12.io.a[2] <= _T_6552 @[TensorGemm.scala 131:22]
    node _T_6553 = asSInt(io.wgt.data.bits[12][2]) @[TensorGemm.scala 132:48]
    dot_12.io.b[2] <= _T_6553 @[TensorGemm.scala 132:22]
    node _T_6554 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_12.io.a[3] <= _T_6554 @[TensorGemm.scala 131:22]
    node _T_6555 = asSInt(io.wgt.data.bits[12][3]) @[TensorGemm.scala 132:48]
    dot_12.io.b[3] <= _T_6555 @[TensorGemm.scala 132:22]
    node _T_6556 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_12.io.a[4] <= _T_6556 @[TensorGemm.scala 131:22]
    node _T_6557 = asSInt(io.wgt.data.bits[12][4]) @[TensorGemm.scala 132:48]
    dot_12.io.b[4] <= _T_6557 @[TensorGemm.scala 132:22]
    node _T_6558 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_12.io.a[5] <= _T_6558 @[TensorGemm.scala 131:22]
    node _T_6559 = asSInt(io.wgt.data.bits[12][5]) @[TensorGemm.scala 132:48]
    dot_12.io.b[5] <= _T_6559 @[TensorGemm.scala 132:22]
    node _T_6560 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_12.io.a[6] <= _T_6560 @[TensorGemm.scala 131:22]
    node _T_6561 = asSInt(io.wgt.data.bits[12][6]) @[TensorGemm.scala 132:48]
    dot_12.io.b[6] <= _T_6561 @[TensorGemm.scala 132:22]
    node _T_6562 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_12.io.a[7] <= _T_6562 @[TensorGemm.scala 131:22]
    node _T_6563 = asSInt(io.wgt.data.bits[12][7]) @[TensorGemm.scala 132:48]
    dot_12.io.b[7] <= _T_6563 @[TensorGemm.scala 132:22]
    node _T_6564 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_12.io.a[8] <= _T_6564 @[TensorGemm.scala 131:22]
    node _T_6565 = asSInt(io.wgt.data.bits[12][8]) @[TensorGemm.scala 132:48]
    dot_12.io.b[8] <= _T_6565 @[TensorGemm.scala 132:22]
    node _T_6566 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_12.io.a[9] <= _T_6566 @[TensorGemm.scala 131:22]
    node _T_6567 = asSInt(io.wgt.data.bits[12][9]) @[TensorGemm.scala 132:48]
    dot_12.io.b[9] <= _T_6567 @[TensorGemm.scala 132:22]
    node _T_6568 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_12.io.a[10] <= _T_6568 @[TensorGemm.scala 131:22]
    node _T_6569 = asSInt(io.wgt.data.bits[12][10]) @[TensorGemm.scala 132:48]
    dot_12.io.b[10] <= _T_6569 @[TensorGemm.scala 132:22]
    node _T_6570 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_12.io.a[11] <= _T_6570 @[TensorGemm.scala 131:22]
    node _T_6571 = asSInt(io.wgt.data.bits[12][11]) @[TensorGemm.scala 132:48]
    dot_12.io.b[11] <= _T_6571 @[TensorGemm.scala 132:22]
    node _T_6572 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_12.io.a[12] <= _T_6572 @[TensorGemm.scala 131:22]
    node _T_6573 = asSInt(io.wgt.data.bits[12][12]) @[TensorGemm.scala 132:48]
    dot_12.io.b[12] <= _T_6573 @[TensorGemm.scala 132:22]
    node _T_6574 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_12.io.a[13] <= _T_6574 @[TensorGemm.scala 131:22]
    node _T_6575 = asSInt(io.wgt.data.bits[12][13]) @[TensorGemm.scala 132:48]
    dot_12.io.b[13] <= _T_6575 @[TensorGemm.scala 132:22]
    node _T_6576 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_12.io.a[14] <= _T_6576 @[TensorGemm.scala 131:22]
    node _T_6577 = asSInt(io.wgt.data.bits[12][14]) @[TensorGemm.scala 132:48]
    dot_12.io.b[14] <= _T_6577 @[TensorGemm.scala 132:22]
    node _T_6578 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_12.io.a[15] <= _T_6578 @[TensorGemm.scala 131:22]
    node _T_6579 = asSInt(io.wgt.data.bits[12][15]) @[TensorGemm.scala 132:48]
    dot_12.io.b[15] <= _T_6579 @[TensorGemm.scala 132:22]
    node _T_6580 = asSInt(acc_12.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6581 = add(_T_6580, dot_12.io.y) @[TensorGemm.scala 134:41]
    node _T_6582 = tail(_T_6581, 1) @[TensorGemm.scala 134:41]
    node _T_6583 = asSInt(_T_6582) @[TensorGemm.scala 134:41]
    add_12 <= _T_6583 @[TensorGemm.scala 134:12]
    node _T_6585 = asUInt(add_12) @[TensorGemm.scala 135:59]
    node _T_6586 = mux(io.reset, UInt<1>("h00"), _T_6585) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][12] <= _T_6586 @[TensorGemm.scala 135:30]
    node _T_6587 = asUInt(add_12) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][12] <= _T_6587 @[TensorGemm.scala 136:28]
    vld[12] <= acc_12.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6588 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6589 = and(_T_6588, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6590 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6591 = and(_T_6589, _T_6590) @[TensorGemm.scala 128:88]
    acc_13.io.enq.valid <= _T_6591 @[TensorGemm.scala 128:25]
    acc_13.io.enq.bits <= io.acc_i.data.bits[0][13] @[TensorGemm.scala 129:24]
    node _T_6592 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_13.io.a[0] <= _T_6592 @[TensorGemm.scala 131:22]
    node _T_6593 = asSInt(io.wgt.data.bits[13][0]) @[TensorGemm.scala 132:48]
    dot_13.io.b[0] <= _T_6593 @[TensorGemm.scala 132:22]
    node _T_6594 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_13.io.a[1] <= _T_6594 @[TensorGemm.scala 131:22]
    node _T_6595 = asSInt(io.wgt.data.bits[13][1]) @[TensorGemm.scala 132:48]
    dot_13.io.b[1] <= _T_6595 @[TensorGemm.scala 132:22]
    node _T_6596 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_13.io.a[2] <= _T_6596 @[TensorGemm.scala 131:22]
    node _T_6597 = asSInt(io.wgt.data.bits[13][2]) @[TensorGemm.scala 132:48]
    dot_13.io.b[2] <= _T_6597 @[TensorGemm.scala 132:22]
    node _T_6598 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_13.io.a[3] <= _T_6598 @[TensorGemm.scala 131:22]
    node _T_6599 = asSInt(io.wgt.data.bits[13][3]) @[TensorGemm.scala 132:48]
    dot_13.io.b[3] <= _T_6599 @[TensorGemm.scala 132:22]
    node _T_6600 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_13.io.a[4] <= _T_6600 @[TensorGemm.scala 131:22]
    node _T_6601 = asSInt(io.wgt.data.bits[13][4]) @[TensorGemm.scala 132:48]
    dot_13.io.b[4] <= _T_6601 @[TensorGemm.scala 132:22]
    node _T_6602 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_13.io.a[5] <= _T_6602 @[TensorGemm.scala 131:22]
    node _T_6603 = asSInt(io.wgt.data.bits[13][5]) @[TensorGemm.scala 132:48]
    dot_13.io.b[5] <= _T_6603 @[TensorGemm.scala 132:22]
    node _T_6604 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_13.io.a[6] <= _T_6604 @[TensorGemm.scala 131:22]
    node _T_6605 = asSInt(io.wgt.data.bits[13][6]) @[TensorGemm.scala 132:48]
    dot_13.io.b[6] <= _T_6605 @[TensorGemm.scala 132:22]
    node _T_6606 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_13.io.a[7] <= _T_6606 @[TensorGemm.scala 131:22]
    node _T_6607 = asSInt(io.wgt.data.bits[13][7]) @[TensorGemm.scala 132:48]
    dot_13.io.b[7] <= _T_6607 @[TensorGemm.scala 132:22]
    node _T_6608 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_13.io.a[8] <= _T_6608 @[TensorGemm.scala 131:22]
    node _T_6609 = asSInt(io.wgt.data.bits[13][8]) @[TensorGemm.scala 132:48]
    dot_13.io.b[8] <= _T_6609 @[TensorGemm.scala 132:22]
    node _T_6610 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_13.io.a[9] <= _T_6610 @[TensorGemm.scala 131:22]
    node _T_6611 = asSInt(io.wgt.data.bits[13][9]) @[TensorGemm.scala 132:48]
    dot_13.io.b[9] <= _T_6611 @[TensorGemm.scala 132:22]
    node _T_6612 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_13.io.a[10] <= _T_6612 @[TensorGemm.scala 131:22]
    node _T_6613 = asSInt(io.wgt.data.bits[13][10]) @[TensorGemm.scala 132:48]
    dot_13.io.b[10] <= _T_6613 @[TensorGemm.scala 132:22]
    node _T_6614 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_13.io.a[11] <= _T_6614 @[TensorGemm.scala 131:22]
    node _T_6615 = asSInt(io.wgt.data.bits[13][11]) @[TensorGemm.scala 132:48]
    dot_13.io.b[11] <= _T_6615 @[TensorGemm.scala 132:22]
    node _T_6616 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_13.io.a[12] <= _T_6616 @[TensorGemm.scala 131:22]
    node _T_6617 = asSInt(io.wgt.data.bits[13][12]) @[TensorGemm.scala 132:48]
    dot_13.io.b[12] <= _T_6617 @[TensorGemm.scala 132:22]
    node _T_6618 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_13.io.a[13] <= _T_6618 @[TensorGemm.scala 131:22]
    node _T_6619 = asSInt(io.wgt.data.bits[13][13]) @[TensorGemm.scala 132:48]
    dot_13.io.b[13] <= _T_6619 @[TensorGemm.scala 132:22]
    node _T_6620 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_13.io.a[14] <= _T_6620 @[TensorGemm.scala 131:22]
    node _T_6621 = asSInt(io.wgt.data.bits[13][14]) @[TensorGemm.scala 132:48]
    dot_13.io.b[14] <= _T_6621 @[TensorGemm.scala 132:22]
    node _T_6622 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_13.io.a[15] <= _T_6622 @[TensorGemm.scala 131:22]
    node _T_6623 = asSInt(io.wgt.data.bits[13][15]) @[TensorGemm.scala 132:48]
    dot_13.io.b[15] <= _T_6623 @[TensorGemm.scala 132:22]
    node _T_6624 = asSInt(acc_13.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6625 = add(_T_6624, dot_13.io.y) @[TensorGemm.scala 134:41]
    node _T_6626 = tail(_T_6625, 1) @[TensorGemm.scala 134:41]
    node _T_6627 = asSInt(_T_6626) @[TensorGemm.scala 134:41]
    add_13 <= _T_6627 @[TensorGemm.scala 134:12]
    node _T_6629 = asUInt(add_13) @[TensorGemm.scala 135:59]
    node _T_6630 = mux(io.reset, UInt<1>("h00"), _T_6629) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][13] <= _T_6630 @[TensorGemm.scala 135:30]
    node _T_6631 = asUInt(add_13) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][13] <= _T_6631 @[TensorGemm.scala 136:28]
    vld[13] <= acc_13.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6632 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6633 = and(_T_6632, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6634 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6635 = and(_T_6633, _T_6634) @[TensorGemm.scala 128:88]
    acc_14.io.enq.valid <= _T_6635 @[TensorGemm.scala 128:25]
    acc_14.io.enq.bits <= io.acc_i.data.bits[0][14] @[TensorGemm.scala 129:24]
    node _T_6636 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_14.io.a[0] <= _T_6636 @[TensorGemm.scala 131:22]
    node _T_6637 = asSInt(io.wgt.data.bits[14][0]) @[TensorGemm.scala 132:48]
    dot_14.io.b[0] <= _T_6637 @[TensorGemm.scala 132:22]
    node _T_6638 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_14.io.a[1] <= _T_6638 @[TensorGemm.scala 131:22]
    node _T_6639 = asSInt(io.wgt.data.bits[14][1]) @[TensorGemm.scala 132:48]
    dot_14.io.b[1] <= _T_6639 @[TensorGemm.scala 132:22]
    node _T_6640 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_14.io.a[2] <= _T_6640 @[TensorGemm.scala 131:22]
    node _T_6641 = asSInt(io.wgt.data.bits[14][2]) @[TensorGemm.scala 132:48]
    dot_14.io.b[2] <= _T_6641 @[TensorGemm.scala 132:22]
    node _T_6642 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_14.io.a[3] <= _T_6642 @[TensorGemm.scala 131:22]
    node _T_6643 = asSInt(io.wgt.data.bits[14][3]) @[TensorGemm.scala 132:48]
    dot_14.io.b[3] <= _T_6643 @[TensorGemm.scala 132:22]
    node _T_6644 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_14.io.a[4] <= _T_6644 @[TensorGemm.scala 131:22]
    node _T_6645 = asSInt(io.wgt.data.bits[14][4]) @[TensorGemm.scala 132:48]
    dot_14.io.b[4] <= _T_6645 @[TensorGemm.scala 132:22]
    node _T_6646 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_14.io.a[5] <= _T_6646 @[TensorGemm.scala 131:22]
    node _T_6647 = asSInt(io.wgt.data.bits[14][5]) @[TensorGemm.scala 132:48]
    dot_14.io.b[5] <= _T_6647 @[TensorGemm.scala 132:22]
    node _T_6648 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_14.io.a[6] <= _T_6648 @[TensorGemm.scala 131:22]
    node _T_6649 = asSInt(io.wgt.data.bits[14][6]) @[TensorGemm.scala 132:48]
    dot_14.io.b[6] <= _T_6649 @[TensorGemm.scala 132:22]
    node _T_6650 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_14.io.a[7] <= _T_6650 @[TensorGemm.scala 131:22]
    node _T_6651 = asSInt(io.wgt.data.bits[14][7]) @[TensorGemm.scala 132:48]
    dot_14.io.b[7] <= _T_6651 @[TensorGemm.scala 132:22]
    node _T_6652 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_14.io.a[8] <= _T_6652 @[TensorGemm.scala 131:22]
    node _T_6653 = asSInt(io.wgt.data.bits[14][8]) @[TensorGemm.scala 132:48]
    dot_14.io.b[8] <= _T_6653 @[TensorGemm.scala 132:22]
    node _T_6654 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_14.io.a[9] <= _T_6654 @[TensorGemm.scala 131:22]
    node _T_6655 = asSInt(io.wgt.data.bits[14][9]) @[TensorGemm.scala 132:48]
    dot_14.io.b[9] <= _T_6655 @[TensorGemm.scala 132:22]
    node _T_6656 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_14.io.a[10] <= _T_6656 @[TensorGemm.scala 131:22]
    node _T_6657 = asSInt(io.wgt.data.bits[14][10]) @[TensorGemm.scala 132:48]
    dot_14.io.b[10] <= _T_6657 @[TensorGemm.scala 132:22]
    node _T_6658 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_14.io.a[11] <= _T_6658 @[TensorGemm.scala 131:22]
    node _T_6659 = asSInt(io.wgt.data.bits[14][11]) @[TensorGemm.scala 132:48]
    dot_14.io.b[11] <= _T_6659 @[TensorGemm.scala 132:22]
    node _T_6660 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_14.io.a[12] <= _T_6660 @[TensorGemm.scala 131:22]
    node _T_6661 = asSInt(io.wgt.data.bits[14][12]) @[TensorGemm.scala 132:48]
    dot_14.io.b[12] <= _T_6661 @[TensorGemm.scala 132:22]
    node _T_6662 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_14.io.a[13] <= _T_6662 @[TensorGemm.scala 131:22]
    node _T_6663 = asSInt(io.wgt.data.bits[14][13]) @[TensorGemm.scala 132:48]
    dot_14.io.b[13] <= _T_6663 @[TensorGemm.scala 132:22]
    node _T_6664 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_14.io.a[14] <= _T_6664 @[TensorGemm.scala 131:22]
    node _T_6665 = asSInt(io.wgt.data.bits[14][14]) @[TensorGemm.scala 132:48]
    dot_14.io.b[14] <= _T_6665 @[TensorGemm.scala 132:22]
    node _T_6666 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_14.io.a[15] <= _T_6666 @[TensorGemm.scala 131:22]
    node _T_6667 = asSInt(io.wgt.data.bits[14][15]) @[TensorGemm.scala 132:48]
    dot_14.io.b[15] <= _T_6667 @[TensorGemm.scala 132:22]
    node _T_6668 = asSInt(acc_14.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6669 = add(_T_6668, dot_14.io.y) @[TensorGemm.scala 134:41]
    node _T_6670 = tail(_T_6669, 1) @[TensorGemm.scala 134:41]
    node _T_6671 = asSInt(_T_6670) @[TensorGemm.scala 134:41]
    add_14 <= _T_6671 @[TensorGemm.scala 134:12]
    node _T_6673 = asUInt(add_14) @[TensorGemm.scala 135:59]
    node _T_6674 = mux(io.reset, UInt<1>("h00"), _T_6673) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][14] <= _T_6674 @[TensorGemm.scala 135:30]
    node _T_6675 = asUInt(add_14) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][14] <= _T_6675 @[TensorGemm.scala 136:28]
    vld[14] <= acc_14.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6676 = and(io.inp.data.valid, io.wgt.data.valid) @[TensorGemm.scala 128:46]
    node _T_6677 = and(_T_6676, io.acc_i.data.valid) @[TensorGemm.scala 128:66]
    node _T_6678 = not(io.reset) @[TensorGemm.scala 128:90]
    node _T_6679 = and(_T_6677, _T_6678) @[TensorGemm.scala 128:88]
    acc_15.io.enq.valid <= _T_6679 @[TensorGemm.scala 128:25]
    acc_15.io.enq.bits <= io.acc_i.data.bits[0][15] @[TensorGemm.scala 129:24]
    node _T_6680 = asSInt(io.inp.data.bits[0][0]) @[TensorGemm.scala 131:48]
    dot_15.io.a[0] <= _T_6680 @[TensorGemm.scala 131:22]
    node _T_6681 = asSInt(io.wgt.data.bits[15][0]) @[TensorGemm.scala 132:48]
    dot_15.io.b[0] <= _T_6681 @[TensorGemm.scala 132:22]
    node _T_6682 = asSInt(io.inp.data.bits[0][1]) @[TensorGemm.scala 131:48]
    dot_15.io.a[1] <= _T_6682 @[TensorGemm.scala 131:22]
    node _T_6683 = asSInt(io.wgt.data.bits[15][1]) @[TensorGemm.scala 132:48]
    dot_15.io.b[1] <= _T_6683 @[TensorGemm.scala 132:22]
    node _T_6684 = asSInt(io.inp.data.bits[0][2]) @[TensorGemm.scala 131:48]
    dot_15.io.a[2] <= _T_6684 @[TensorGemm.scala 131:22]
    node _T_6685 = asSInt(io.wgt.data.bits[15][2]) @[TensorGemm.scala 132:48]
    dot_15.io.b[2] <= _T_6685 @[TensorGemm.scala 132:22]
    node _T_6686 = asSInt(io.inp.data.bits[0][3]) @[TensorGemm.scala 131:48]
    dot_15.io.a[3] <= _T_6686 @[TensorGemm.scala 131:22]
    node _T_6687 = asSInt(io.wgt.data.bits[15][3]) @[TensorGemm.scala 132:48]
    dot_15.io.b[3] <= _T_6687 @[TensorGemm.scala 132:22]
    node _T_6688 = asSInt(io.inp.data.bits[0][4]) @[TensorGemm.scala 131:48]
    dot_15.io.a[4] <= _T_6688 @[TensorGemm.scala 131:22]
    node _T_6689 = asSInt(io.wgt.data.bits[15][4]) @[TensorGemm.scala 132:48]
    dot_15.io.b[4] <= _T_6689 @[TensorGemm.scala 132:22]
    node _T_6690 = asSInt(io.inp.data.bits[0][5]) @[TensorGemm.scala 131:48]
    dot_15.io.a[5] <= _T_6690 @[TensorGemm.scala 131:22]
    node _T_6691 = asSInt(io.wgt.data.bits[15][5]) @[TensorGemm.scala 132:48]
    dot_15.io.b[5] <= _T_6691 @[TensorGemm.scala 132:22]
    node _T_6692 = asSInt(io.inp.data.bits[0][6]) @[TensorGemm.scala 131:48]
    dot_15.io.a[6] <= _T_6692 @[TensorGemm.scala 131:22]
    node _T_6693 = asSInt(io.wgt.data.bits[15][6]) @[TensorGemm.scala 132:48]
    dot_15.io.b[6] <= _T_6693 @[TensorGemm.scala 132:22]
    node _T_6694 = asSInt(io.inp.data.bits[0][7]) @[TensorGemm.scala 131:48]
    dot_15.io.a[7] <= _T_6694 @[TensorGemm.scala 131:22]
    node _T_6695 = asSInt(io.wgt.data.bits[15][7]) @[TensorGemm.scala 132:48]
    dot_15.io.b[7] <= _T_6695 @[TensorGemm.scala 132:22]
    node _T_6696 = asSInt(io.inp.data.bits[0][8]) @[TensorGemm.scala 131:48]
    dot_15.io.a[8] <= _T_6696 @[TensorGemm.scala 131:22]
    node _T_6697 = asSInt(io.wgt.data.bits[15][8]) @[TensorGemm.scala 132:48]
    dot_15.io.b[8] <= _T_6697 @[TensorGemm.scala 132:22]
    node _T_6698 = asSInt(io.inp.data.bits[0][9]) @[TensorGemm.scala 131:48]
    dot_15.io.a[9] <= _T_6698 @[TensorGemm.scala 131:22]
    node _T_6699 = asSInt(io.wgt.data.bits[15][9]) @[TensorGemm.scala 132:48]
    dot_15.io.b[9] <= _T_6699 @[TensorGemm.scala 132:22]
    node _T_6700 = asSInt(io.inp.data.bits[0][10]) @[TensorGemm.scala 131:48]
    dot_15.io.a[10] <= _T_6700 @[TensorGemm.scala 131:22]
    node _T_6701 = asSInt(io.wgt.data.bits[15][10]) @[TensorGemm.scala 132:48]
    dot_15.io.b[10] <= _T_6701 @[TensorGemm.scala 132:22]
    node _T_6702 = asSInt(io.inp.data.bits[0][11]) @[TensorGemm.scala 131:48]
    dot_15.io.a[11] <= _T_6702 @[TensorGemm.scala 131:22]
    node _T_6703 = asSInt(io.wgt.data.bits[15][11]) @[TensorGemm.scala 132:48]
    dot_15.io.b[11] <= _T_6703 @[TensorGemm.scala 132:22]
    node _T_6704 = asSInt(io.inp.data.bits[0][12]) @[TensorGemm.scala 131:48]
    dot_15.io.a[12] <= _T_6704 @[TensorGemm.scala 131:22]
    node _T_6705 = asSInt(io.wgt.data.bits[15][12]) @[TensorGemm.scala 132:48]
    dot_15.io.b[12] <= _T_6705 @[TensorGemm.scala 132:22]
    node _T_6706 = asSInt(io.inp.data.bits[0][13]) @[TensorGemm.scala 131:48]
    dot_15.io.a[13] <= _T_6706 @[TensorGemm.scala 131:22]
    node _T_6707 = asSInt(io.wgt.data.bits[15][13]) @[TensorGemm.scala 132:48]
    dot_15.io.b[13] <= _T_6707 @[TensorGemm.scala 132:22]
    node _T_6708 = asSInt(io.inp.data.bits[0][14]) @[TensorGemm.scala 131:48]
    dot_15.io.a[14] <= _T_6708 @[TensorGemm.scala 131:22]
    node _T_6709 = asSInt(io.wgt.data.bits[15][14]) @[TensorGemm.scala 132:48]
    dot_15.io.b[14] <= _T_6709 @[TensorGemm.scala 132:22]
    node _T_6710 = asSInt(io.inp.data.bits[0][15]) @[TensorGemm.scala 131:48]
    dot_15.io.a[15] <= _T_6710 @[TensorGemm.scala 131:22]
    node _T_6711 = asSInt(io.wgt.data.bits[15][15]) @[TensorGemm.scala 132:48]
    dot_15.io.b[15] <= _T_6711 @[TensorGemm.scala 132:22]
    node _T_6712 = asSInt(acc_15.io.deq.bits) @[TensorGemm.scala 134:34]
    node _T_6713 = add(_T_6712, dot_15.io.y) @[TensorGemm.scala 134:41]
    node _T_6714 = tail(_T_6713, 1) @[TensorGemm.scala 134:41]
    node _T_6715 = asSInt(_T_6714) @[TensorGemm.scala 134:41]
    add_15 <= _T_6715 @[TensorGemm.scala 134:12]
    node _T_6717 = asUInt(add_15) @[TensorGemm.scala 135:59]
    node _T_6718 = mux(io.reset, UInt<1>("h00"), _T_6717) @[TensorGemm.scala 135:36]
    io.acc_o.data.bits[0][15] <= _T_6718 @[TensorGemm.scala 135:30]
    node _T_6719 = asUInt(add_15) @[TensorGemm.scala 136:38]
    io.out.data.bits[0][15] <= _T_6719 @[TensorGemm.scala 136:28]
    vld[15] <= acc_15.io.deq.valid @[TensorGemm.scala 137:12]
    node _T_6720 = cat(vld[1], vld[0]) @[TensorGemm.scala 139:30]
    node _T_6721 = cat(vld[3], vld[2]) @[TensorGemm.scala 139:30]
    node _T_6722 = cat(_T_6721, _T_6720) @[TensorGemm.scala 139:30]
    node _T_6723 = cat(vld[5], vld[4]) @[TensorGemm.scala 139:30]
    node _T_6724 = cat(vld[7], vld[6]) @[TensorGemm.scala 139:30]
    node _T_6725 = cat(_T_6724, _T_6723) @[TensorGemm.scala 139:30]
    node _T_6726 = cat(_T_6725, _T_6722) @[TensorGemm.scala 139:30]
    node _T_6727 = cat(vld[9], vld[8]) @[TensorGemm.scala 139:30]
    node _T_6728 = cat(vld[11], vld[10]) @[TensorGemm.scala 139:30]
    node _T_6729 = cat(_T_6728, _T_6727) @[TensorGemm.scala 139:30]
    node _T_6730 = cat(vld[13], vld[12]) @[TensorGemm.scala 139:30]
    node _T_6731 = cat(vld[15], vld[14]) @[TensorGemm.scala 139:30]
    node _T_6732 = cat(_T_6731, _T_6730) @[TensorGemm.scala 139:30]
    node _T_6733 = cat(_T_6732, _T_6729) @[TensorGemm.scala 139:30]
    node _T_6734 = cat(_T_6733, _T_6726) @[TensorGemm.scala 139:30]
    node _T_6735 = not(_T_6734) @[TensorGemm.scala 139:37]
    node _T_6737 = eq(_T_6735, UInt<1>("h00")) @[TensorGemm.scala 139:37]
    node _T_6738 = or(_T_6737, io.reset) @[TensorGemm.scala 139:42]
    io.acc_o.data.valid <= _T_6738 @[TensorGemm.scala 139:23]
    node _T_6739 = cat(vld[1], vld[0]) @[TensorGemm.scala 140:28]
    node _T_6740 = cat(vld[3], vld[2]) @[TensorGemm.scala 140:28]
    node _T_6741 = cat(_T_6740, _T_6739) @[TensorGemm.scala 140:28]
    node _T_6742 = cat(vld[5], vld[4]) @[TensorGemm.scala 140:28]
    node _T_6743 = cat(vld[7], vld[6]) @[TensorGemm.scala 140:28]
    node _T_6744 = cat(_T_6743, _T_6742) @[TensorGemm.scala 140:28]
    node _T_6745 = cat(_T_6744, _T_6741) @[TensorGemm.scala 140:28]
    node _T_6746 = cat(vld[9], vld[8]) @[TensorGemm.scala 140:28]
    node _T_6747 = cat(vld[11], vld[10]) @[TensorGemm.scala 140:28]
    node _T_6748 = cat(_T_6747, _T_6746) @[TensorGemm.scala 140:28]
    node _T_6749 = cat(vld[13], vld[12]) @[TensorGemm.scala 140:28]
    node _T_6750 = cat(vld[15], vld[14]) @[TensorGemm.scala 140:28]
    node _T_6751 = cat(_T_6750, _T_6749) @[TensorGemm.scala 140:28]
    node _T_6752 = cat(_T_6751, _T_6748) @[TensorGemm.scala 140:28]
    node _T_6753 = cat(_T_6752, _T_6745) @[TensorGemm.scala 140:28]
    node _T_6754 = not(_T_6753) @[TensorGemm.scala 140:35]
    node _T_6756 = eq(_T_6754, UInt<1>("h00")) @[TensorGemm.scala 140:35]
    io.out.data.valid <= _T_6756 @[TensorGemm.scala 140:21]
    
