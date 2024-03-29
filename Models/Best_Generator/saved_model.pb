��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8ۿ
�
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameconv2d_transpose_7/kernel
�
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*'
_output_shapes
:�*
dtype0
�
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_11/moving_variance
�
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_11/moving_mean
�
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_11/beta
�
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_11/gamma
�
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameconv2d_transpose_6/bias
�
+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_nameconv2d_transpose_6/kernel
�
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*(
_output_shapes
:��*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameconv2d_transpose_5/bias
�
+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_nameconv2d_transpose_5/kernel
�
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameconv2d_transpose_4/bias
�
+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_nameconv2d_transpose_4/kernel
�
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*(
_output_shapes
:��*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:� *
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�� *
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
�� *
dtype0
�
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameembedding_2/embeddings
�
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	�*
dtype0
|
serving_default_input_5Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
z
serving_default_input_6Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_6embedding_2/embeddingsdense_3/kerneldense_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_transpose_6/kernelconv2d_transpose_6/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_transpose_7/kernelconv2d_transpose_7/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_691043710

NoOpNoOp
�h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�h
value�hB�h B�h
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!
embeddings*
* 
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance*
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

vkernel
wbias
 x_jit_compiled_convolution_op*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
!0
41
52
B3
C4
R5
S6
T7
U8
\9
]10
l11
m12
n13
o14
v15
w16
�17
�18
�19
�20
�21
�22*
�
!0
41
52
B3
C4
R5
S6
\7
]8
l9
m10
v11
w12
�13
�14
�15
�16*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

!0*

!0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

B0
C1*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
R0
S1
T2
U3*

R0
S1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
l0
m1
n2
o3*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

v0
w1*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_7/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_7/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
0
T0
U1
n2
o3
�4
�5*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

T0
U1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

n0
o1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_2/embeddings/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp-conv2d_transpose_7/kernel/Read/ReadVariableOp+conv2d_transpose_7/bias/Read/ReadVariableOpConst*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_save_691044686
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_3/kerneldense_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_transpose_6/kernelconv2d_transpose_6/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_transpose_7/kernelconv2d_transpose_7/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference__traced_restore_691044765��
�
�
+__inference_dense_3_layer_call_fn_691044170

inputs
unknown:
�� 
	unknown_0:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_11_layer_call_fn_691044505

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042986�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_10_layer_call_fn_691044378

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042847�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
I
-__inference_reshape_1_layer_call_fn_691044185

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:���������� :P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
6__inference_conv2d_transpose_6_layer_call_fn_691044436

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042739

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044541

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
6__inference_conv2d_transpose_7_layer_call_fn_691044550

inputs"
unknown:�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042878

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
x
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691044161
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�P
�
H__inference_generator_layer_call_and_return_conditional_losses_691043421

inputs
inputs_1(
embedding_2_691043358:	�%
dense_3_691043363:
��  
dense_3_691043365:	� 8
conv2d_transpose_4_691043369:��+
conv2d_transpose_4_691043371:	�.
batch_normalization_9_691043375:	�.
batch_normalization_9_691043377:	�.
batch_normalization_9_691043379:	�.
batch_normalization_9_691043381:	�8
conv2d_transpose_5_691043384:��+
conv2d_transpose_5_691043386:	�/
 batch_normalization_10_691043390:	�/
 batch_normalization_10_691043392:	�/
 batch_normalization_10_691043394:	�/
 batch_normalization_10_691043396:	�8
conv2d_transpose_6_691043399:��+
conv2d_transpose_6_691043401:	�/
 batch_normalization_11_691043405:	�/
 batch_normalization_11_691043407:	�/
 batch_normalization_11_691043409:	�/
 batch_normalization_11_691043411:	�7
conv2d_transpose_7_691043414:�*
conv2d_transpose_7_691043416:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_2_691043358*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060�
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070�
concatenate_2/PartitionedCallPartitionedCallinputs"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_3_691043363dense_3_691043365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091�
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_4_691043369conv2d_transpose_4_691043371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710�
leaky_re_lu_10/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0batch_normalization_9_691043375batch_normalization_9_691043377batch_normalization_9_691043379batch_normalization_9_691043381*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042770�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_transpose_5_691043384conv2d_transpose_5_691043386*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818�
leaky_re_lu_11/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0 batch_normalization_10_691043390 batch_normalization_10_691043392 batch_normalization_10_691043394 batch_normalization_10_691043396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042878�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_transpose_6_691043399conv2d_transpose_6_691043401*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926�
leaky_re_lu_12/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0 batch_normalization_11_691043405 batch_normalization_11_691043407 batch_normalization_11_691043409 batch_normalization_11_691043411*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042986�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_7_691043414conv2d_transpose_7_691043416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034�
activation_1/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691044479

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:���������  �h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
L
0__inference_activation_1_layer_call_fn_691044588

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�P
�
H__inference_generator_layer_call_and_return_conditional_losses_691043589
input_5
input_6(
embedding_2_691043526:	�%
dense_3_691043531:
��  
dense_3_691043533:	� 8
conv2d_transpose_4_691043537:��+
conv2d_transpose_4_691043539:	�.
batch_normalization_9_691043543:	�.
batch_normalization_9_691043545:	�.
batch_normalization_9_691043547:	�.
batch_normalization_9_691043549:	�8
conv2d_transpose_5_691043552:��+
conv2d_transpose_5_691043554:	�/
 batch_normalization_10_691043558:	�/
 batch_normalization_10_691043560:	�/
 batch_normalization_10_691043562:	�/
 batch_normalization_10_691043564:	�8
conv2d_transpose_6_691043567:��+
conv2d_transpose_6_691043569:	�/
 batch_normalization_11_691043573:	�/
 batch_normalization_11_691043575:	�/
 batch_normalization_11_691043577:	�/
 batch_normalization_11_691043579:	�7
conv2d_transpose_7_691043582:�*
conv2d_transpose_7_691043584:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_6embedding_2_691043526*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060�
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070�
concatenate_2/PartitionedCallPartitionedCallinput_5"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_3_691043531dense_3_691043533*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091�
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_4_691043537conv2d_transpose_4_691043539*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710�
leaky_re_lu_10/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0batch_normalization_9_691043543batch_normalization_9_691043545batch_normalization_9_691043547batch_normalization_9_691043549*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042739�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_transpose_5_691043552conv2d_transpose_5_691043554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818�
leaky_re_lu_11/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0 batch_normalization_10_691043558 batch_normalization_10_691043560 batch_normalization_10_691043562 batch_normalization_10_691043564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042847�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_transpose_6_691043567conv2d_transpose_6_691043569*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926�
leaky_re_lu_12/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0 batch_normalization_11_691043573 batch_normalization_11_691043575 batch_normalization_11_691043577 batch_normalization_11_691043579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042955�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_7_691043582conv2d_transpose_7_691043584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034�
activation_1/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
� 
�
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691044241

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
/__inference_embedding_2_layer_call_fn_691044127

inputs
unknown:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
1__inference_concatenate_2_layer_call_fn_691044154
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�	
�
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060

inputs-
embedding_lookup_691043054:	�
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_691043054Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/691043054*,
_output_shapes
:����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/691043054*,
_output_shapes
:�����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_generator_layer_call_fn_691043814
inputs_0
inputs_1
unknown:	�
	unknown_0:
�� 
	unknown_1:	� %
	unknown_2:��
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�%
	unknown_8:��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�&

unknown_14:��

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:�

unknown_21:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*3
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_generator_layer_call_and_return_conditional_losses_691043421w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
6__inference_conv2d_transpose_5_layer_call_fn_691044322

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
H__inference_generator_layer_call_and_return_conditional_losses_691044120
inputs_0
inputs_19
&embedding_2_embedding_lookup_691043972:	�:
&dense_3_matmul_readvariableop_resource:
�� 6
'dense_3_biasadd_readvariableop_resource:	� W
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�W
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_5_biasadd_readvariableop_resource:	�=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�W
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_6_biasadd_readvariableop_resource:	�=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�V
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:�@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��%batch_normalization_10/AssignNewValue�'batch_normalization_10/AssignNewValue_1�6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�%batch_normalization_11/AssignNewValue�'batch_normalization_11/AssignNewValue_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_2/embedding_lookupc
embedding_2/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_2/embedding_lookupResourceGather&embedding_2_embedding_lookup_691043972embedding_2/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_2/embedding_lookup/691043972*,
_output_shapes
:����������*
dtype0�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_2/embedding_lookup/691043972*,
_output_shapes
:�����������
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_3/ReshapeReshape0embedding_2/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2inputs_0flatten_3/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
dense_3/MatMulMatMulconcatenate_2/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� W
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������b
conv2d_transpose_4/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������|
leaky_re_lu_10/LeakyRelu	LeakyRelu#conv2d_transpose_4/BiasAdd:output:0*0
_output_shapes
:�����������
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_10/LeakyRelu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
conv2d_transpose_5/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������|
leaky_re_lu_11/LeakyRelu	LeakyRelu#conv2d_transpose_5/BiasAdd:output:0*0
_output_shapes
:�����������
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_11/LeakyRelu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
conv2d_transpose_6/ShapeShape+batch_normalization_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �|
leaky_re_lu_12/LeakyRelu	LeakyRelu#conv2d_transpose_6/BiasAdd:output:0*0
_output_shapes
:���������  ��
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_12/LeakyRelu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
conv2d_transpose_7/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:�*
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@x
activation_1/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@l
IdentityIdentityactivation_1/Tanh:y:0^NoOp*
T0*/
_output_shapes
:���������@@�

NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_2/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
F__inference_dense_3_layer_call_and_return_conditional_losses_691044180

inputs2
matmul_readvariableop_resource:
�� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:���������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_flatten_3_layer_call_fn_691044142

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_leaky_re_lu_11_layer_call_fn_691044360

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_9_layer_call_fn_691044277

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042770�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�`
�
%__inference__traced_restore_691044765
file_prefix:
'assignvariableop_embedding_2_embeddings:	�5
!assignvariableop_1_dense_3_kernel:
�� .
assignvariableop_2_dense_3_bias:	� H
,assignvariableop_3_conv2d_transpose_4_kernel:��9
*assignvariableop_4_conv2d_transpose_4_bias:	�=
.assignvariableop_5_batch_normalization_9_gamma:	�<
-assignvariableop_6_batch_normalization_9_beta:	�C
4assignvariableop_7_batch_normalization_9_moving_mean:	�G
8assignvariableop_8_batch_normalization_9_moving_variance:	�H
,assignvariableop_9_conv2d_transpose_5_kernel:��:
+assignvariableop_10_conv2d_transpose_5_bias:	�?
0assignvariableop_11_batch_normalization_10_gamma:	�>
/assignvariableop_12_batch_normalization_10_beta:	�E
6assignvariableop_13_batch_normalization_10_moving_mean:	�I
:assignvariableop_14_batch_normalization_10_moving_variance:	�I
-assignvariableop_15_conv2d_transpose_6_kernel:��:
+assignvariableop_16_conv2d_transpose_6_bias:	�?
0assignvariableop_17_batch_normalization_11_gamma:	�>
/assignvariableop_18_batch_normalization_11_beta:	�E
6assignvariableop_19_batch_normalization_11_moving_mean:	�I
:assignvariableop_20_batch_normalization_11_moving_variance:	�H
-assignvariableop_21_conv2d_transpose_7_kernel:�9
+assignvariableop_22_conv2d_transpose_7_bias:
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_3_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_3_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_conv2d_transpose_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv2d_transpose_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_9_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_batch_normalization_9_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp4assignvariableop_7_batch_normalization_9_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_batch_normalization_9_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_conv2d_transpose_5_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_conv2d_transpose_5_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_10_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_10_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_10_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_10_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_conv2d_transpose_6_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_conv2d_transpose_6_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_11_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_11_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_11_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_11_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_7_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_7_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
� 
�
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
N
2__inference_leaky_re_lu_12_layer_call_fn_691044474

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
d
H__inference_flatten_3_layer_call_and_return_conditional_losses_691044148

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_embedding_2_layer_call_and_return_conditional_losses_691044137

inputs-
embedding_lookup_691044131:	�
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_691044131Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/691044131*,
_output_shapes
:����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/691044131*,
_output_shapes
:�����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691044583

inputsC
(conv2d_transpose_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
� 
�
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034

inputsC
(conv2d_transpose_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044313

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:���������� :P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044427

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_9_layer_call_fn_691044264

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042739�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
N
2__inference_leaky_re_lu_10_layer_call_fn_691044246

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_generator_layer_call_fn_691043238
input_5
input_6
unknown:	�
	unknown_0:
�� 
	unknown_1:	� %
	unknown_2:��
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�%
	unknown_8:��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�&

unknown_14:��

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:�

unknown_21:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_generator_layer_call_and_return_conditional_losses_691043189w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044409

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_reshape_1_layer_call_and_return_conditional_losses_691044199

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:���������� :P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042986

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
-__inference_generator_layer_call_fn_691043762
inputs_0
inputs_1
unknown:	�
	unknown_0:
�� 
	unknown_1:	� %
	unknown_2:��
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�%
	unknown_8:��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�&

unknown_14:��

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:�

unknown_21:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_generator_layer_call_and_return_conditional_losses_691043189w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042847

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_11_layer_call_fn_691044492

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042955�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186

inputs
identityN
TanhTanhinputs*
T0*/
_output_shapes
:���������@@X
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_10_layer_call_fn_691044391

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042878�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
i
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691044251

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�P
�
H__inference_generator_layer_call_and_return_conditional_losses_691043189

inputs
inputs_1(
embedding_2_691043061:	�%
dense_3_691043092:
��  
dense_3_691043094:	� 8
conv2d_transpose_4_691043113:��+
conv2d_transpose_4_691043115:	�.
batch_normalization_9_691043125:	�.
batch_normalization_9_691043127:	�.
batch_normalization_9_691043129:	�.
batch_normalization_9_691043131:	�8
conv2d_transpose_5_691043134:��+
conv2d_transpose_5_691043136:	�/
 batch_normalization_10_691043146:	�/
 batch_normalization_10_691043148:	�/
 batch_normalization_10_691043150:	�/
 batch_normalization_10_691043152:	�8
conv2d_transpose_6_691043155:��+
conv2d_transpose_6_691043157:	�/
 batch_normalization_11_691043167:	�/
 batch_normalization_11_691043169:	�/
 batch_normalization_11_691043171:	�/
 batch_normalization_11_691043173:	�7
conv2d_transpose_7_691043176:�*
conv2d_transpose_7_691043178:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_2_691043061*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060�
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070�
concatenate_2/PartitionedCallPartitionedCallinputs"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_3_691043092dense_3_691043094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091�
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_4_691043113conv2d_transpose_4_691043115*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710�
leaky_re_lu_10/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0batch_normalization_9_691043125batch_normalization_9_691043127batch_normalization_9_691043129batch_normalization_9_691043131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042739�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_transpose_5_691043134conv2d_transpose_5_691043136*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818�
leaky_re_lu_11/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0 batch_normalization_10_691043146 batch_normalization_10_691043148 batch_normalization_10_691043150 batch_normalization_10_691043152*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042847�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_transpose_6_691043155conv2d_transpose_6_691043157*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926�
leaky_re_lu_12/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0 batch_normalization_11_691043167 batch_normalization_11_691043169 batch_normalization_11_691043171 batch_normalization_11_691043173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042955�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_7_691043176conv2d_transpose_7_691043178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034�
activation_1/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
H__inference_generator_layer_call_and_return_conditional_losses_691043967
inputs_0
inputs_19
&embedding_2_embedding_lookup_691043819:	�:
&dense_3_matmul_readvariableop_resource:
�� 6
'dense_3_biasadd_readvariableop_resource:	� W
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�W
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_5_biasadd_readvariableop_resource:	�=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�W
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_6_biasadd_readvariableop_resource:	�=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�V
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:�@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_2/embedding_lookupc
embedding_2/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_2/embedding_lookupResourceGather&embedding_2_embedding_lookup_691043819embedding_2/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_2/embedding_lookup/691043819*,
_output_shapes
:����������*
dtype0�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_2/embedding_lookup/691043819*,
_output_shapes
:�����������
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_3/ReshapeReshape0embedding_2/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2inputs_0flatten_3/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
dense_3/MatMulMatMulconcatenate_2/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� W
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������b
conv2d_transpose_4/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������|
leaky_re_lu_10/LeakyRelu	LeakyRelu#conv2d_transpose_4/BiasAdd:output:0*0
_output_shapes
:�����������
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_10/LeakyRelu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
is_training( r
conv2d_transpose_5/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������|
leaky_re_lu_11/LeakyRelu	LeakyRelu#conv2d_transpose_5/BiasAdd:output:0*0
_output_shapes
:�����������
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_11/LeakyRelu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
is_training( s
conv2d_transpose_6/ShapeShape+batch_normalization_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �|
leaky_re_lu_12/LeakyRelu	LeakyRelu#conv2d_transpose_6/BiasAdd:output:0*0
_output_shapes
:���������  ��
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_12/LeakyRelu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%��'7*
is_training( s
conv2d_transpose_7/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:�*
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@x
activation_1/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@l
IdentityIdentityactivation_1/Tanh:y:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_2/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042955

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091

inputs2
matmul_readvariableop_resource:
�� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:���������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
$__inference__wrapped_model_691042673
input_5
input_6C
0generator_embedding_2_embedding_lookup_691042525:	�D
0generator_dense_3_matmul_readvariableop_resource:
�� @
1generator_dense_3_biasadd_readvariableop_resource:	� a
Egenerator_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��K
<generator_conv2d_transpose_4_biasadd_readvariableop_resource:	�F
7generator_batch_normalization_9_readvariableop_resource:	�H
9generator_batch_normalization_9_readvariableop_1_resource:	�W
Hgenerator_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�Y
Jgenerator_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�a
Egenerator_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:��K
<generator_conv2d_transpose_5_biasadd_readvariableop_resource:	�G
8generator_batch_normalization_10_readvariableop_resource:	�I
:generator_batch_normalization_10_readvariableop_1_resource:	�X
Igenerator_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�Z
Kgenerator_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�a
Egenerator_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:��K
<generator_conv2d_transpose_6_biasadd_readvariableop_resource:	�G
8generator_batch_normalization_11_readvariableop_resource:	�I
:generator_batch_normalization_11_readvariableop_1_resource:	�X
Igenerator_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�Z
Kgenerator_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�`
Egenerator_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:�J
<generator_conv2d_transpose_7_biasadd_readvariableop_resource:
identity��@generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp�Bgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�/generator/batch_normalization_10/ReadVariableOp�1generator/batch_normalization_10/ReadVariableOp_1�@generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp�Bgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�/generator/batch_normalization_11/ReadVariableOp�1generator/batch_normalization_11/ReadVariableOp_1�?generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Agenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�.generator/batch_normalization_9/ReadVariableOp�0generator/batch_normalization_9/ReadVariableOp_1�3generator/conv2d_transpose_4/BiasAdd/ReadVariableOp�<generator/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�3generator/conv2d_transpose_5/BiasAdd/ReadVariableOp�<generator/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�3generator/conv2d_transpose_6/BiasAdd/ReadVariableOp�<generator/conv2d_transpose_6/conv2d_transpose/ReadVariableOp�3generator/conv2d_transpose_7/BiasAdd/ReadVariableOp�<generator/conv2d_transpose_7/conv2d_transpose/ReadVariableOp�(generator/dense_3/BiasAdd/ReadVariableOp�'generator/dense_3/MatMul/ReadVariableOp�&generator/embedding_2/embedding_lookupl
generator/embedding_2/CastCastinput_6*

DstT0*

SrcT0*'
_output_shapes
:����������
&generator/embedding_2/embedding_lookupResourceGather0generator_embedding_2_embedding_lookup_691042525generator/embedding_2/Cast:y:0*
Tindices0*C
_class9
75loc:@generator/embedding_2/embedding_lookup/691042525*,
_output_shapes
:����������*
dtype0�
/generator/embedding_2/embedding_lookup/IdentityIdentity/generator/embedding_2/embedding_lookup:output:0*
T0*C
_class9
75loc:@generator/embedding_2/embedding_lookup/691042525*,
_output_shapes
:�����������
1generator/embedding_2/embedding_lookup/Identity_1Identity8generator/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������j
generator/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
generator/flatten_3/ReshapeReshape:generator/embedding_2/embedding_lookup/Identity_1:output:0"generator/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������e
#generator/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
generator/concatenate_2/concatConcatV2input_5$generator/flatten_3/Reshape:output:0,generator/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
'generator/dense_3/MatMul/ReadVariableOpReadVariableOp0generator_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
generator/dense_3/MatMulMatMul'generator/concatenate_2/concat:output:0/generator/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
(generator/dense_3/BiasAdd/ReadVariableOpReadVariableOp1generator_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
generator/dense_3/BiasAddBiasAdd"generator/dense_3/MatMul:product:00generator/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� k
generator/reshape_1/ShapeShape"generator/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:q
'generator/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)generator/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)generator/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!generator/reshape_1/strided_sliceStridedSlice"generator/reshape_1/Shape:output:00generator/reshape_1/strided_slice/stack:output:02generator/reshape_1/strided_slice/stack_1:output:02generator/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#generator/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#generator/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :f
#generator/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
!generator/reshape_1/Reshape/shapePack*generator/reshape_1/strided_slice:output:0,generator/reshape_1/Reshape/shape/1:output:0,generator/reshape_1/Reshape/shape/2:output:0,generator/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
generator/reshape_1/ReshapeReshape"generator/dense_3/BiasAdd:output:0*generator/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������v
"generator/conv2d_transpose_4/ShapeShape$generator/reshape_1/Reshape:output:0*
T0*
_output_shapes
:z
0generator/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2generator/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2generator/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*generator/conv2d_transpose_4/strided_sliceStridedSlice+generator/conv2d_transpose_4/Shape:output:09generator/conv2d_transpose_4/strided_slice/stack:output:0;generator/conv2d_transpose_4/strided_slice/stack_1:output:0;generator/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$generator/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :f
$generator/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :g
$generator/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
"generator/conv2d_transpose_4/stackPack3generator/conv2d_transpose_4/strided_slice:output:0-generator/conv2d_transpose_4/stack/1:output:0-generator/conv2d_transpose_4/stack/2:output:0-generator/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:|
2generator/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4generator/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4generator/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,generator/conv2d_transpose_4/strided_slice_1StridedSlice+generator/conv2d_transpose_4/stack:output:0;generator/conv2d_transpose_4/strided_slice_1/stack:output:0=generator/conv2d_transpose_4/strided_slice_1/stack_1:output:0=generator/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
<generator/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpEgenerator_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
-generator/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput+generator/conv2d_transpose_4/stack:output:0Dgenerator/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0$generator/reshape_1/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
3generator/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp<generator_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$generator/conv2d_transpose_4/BiasAddBiasAdd6generator/conv2d_transpose_4/conv2d_transpose:output:0;generator/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"generator/leaky_re_lu_10/LeakyRelu	LeakyRelu-generator/conv2d_transpose_4/BiasAdd:output:0*0
_output_shapes
:�����������
.generator/batch_normalization_9/ReadVariableOpReadVariableOp7generator_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0generator/batch_normalization_9/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
?generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Agenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0generator/batch_normalization_9/FusedBatchNormV3FusedBatchNormV30generator/leaky_re_lu_10/LeakyRelu:activations:06generator/batch_normalization_9/ReadVariableOp:value:08generator/batch_normalization_9/ReadVariableOp_1:value:0Ggenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Igenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
is_training( �
"generator/conv2d_transpose_5/ShapeShape4generator/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:z
0generator/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2generator/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2generator/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*generator/conv2d_transpose_5/strided_sliceStridedSlice+generator/conv2d_transpose_5/Shape:output:09generator/conv2d_transpose_5/strided_slice/stack:output:0;generator/conv2d_transpose_5/strided_slice/stack_1:output:0;generator/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$generator/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :f
$generator/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :g
$generator/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
"generator/conv2d_transpose_5/stackPack3generator/conv2d_transpose_5/strided_slice:output:0-generator/conv2d_transpose_5/stack/1:output:0-generator/conv2d_transpose_5/stack/2:output:0-generator/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:|
2generator/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4generator/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4generator/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,generator/conv2d_transpose_5/strided_slice_1StridedSlice+generator/conv2d_transpose_5/stack:output:0;generator/conv2d_transpose_5/strided_slice_1/stack:output:0=generator/conv2d_transpose_5/strided_slice_1/stack_1:output:0=generator/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
<generator/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpEgenerator_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
-generator/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput+generator/conv2d_transpose_5/stack:output:0Dgenerator/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:04generator/batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
3generator/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp<generator_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$generator/conv2d_transpose_5/BiasAddBiasAdd6generator/conv2d_transpose_5/conv2d_transpose:output:0;generator/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"generator/leaky_re_lu_11/LeakyRelu	LeakyRelu-generator/conv2d_transpose_5/BiasAdd:output:0*0
_output_shapes
:�����������
/generator/batch_normalization_10/ReadVariableOpReadVariableOp8generator_batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1generator/batch_normalization_10/ReadVariableOp_1ReadVariableOp:generator_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
@generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpIgenerator_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Bgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKgenerator_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
1generator/batch_normalization_10/FusedBatchNormV3FusedBatchNormV30generator/leaky_re_lu_11/LeakyRelu:activations:07generator/batch_normalization_10/ReadVariableOp:value:09generator/batch_normalization_10/ReadVariableOp_1:value:0Hgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Jgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%��'7*
is_training( �
"generator/conv2d_transpose_6/ShapeShape5generator/batch_normalization_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:z
0generator/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2generator/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2generator/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*generator/conv2d_transpose_6/strided_sliceStridedSlice+generator/conv2d_transpose_6/Shape:output:09generator/conv2d_transpose_6/strided_slice/stack:output:0;generator/conv2d_transpose_6/strided_slice/stack_1:output:0;generator/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$generator/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : f
$generator/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B : g
$generator/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
"generator/conv2d_transpose_6/stackPack3generator/conv2d_transpose_6/strided_slice:output:0-generator/conv2d_transpose_6/stack/1:output:0-generator/conv2d_transpose_6/stack/2:output:0-generator/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:|
2generator/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4generator/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4generator/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,generator/conv2d_transpose_6/strided_slice_1StridedSlice+generator/conv2d_transpose_6/stack:output:0;generator/conv2d_transpose_6/strided_slice_1/stack:output:0=generator/conv2d_transpose_6/strided_slice_1/stack_1:output:0=generator/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
<generator/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpEgenerator_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
-generator/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput+generator/conv2d_transpose_6/stack:output:0Dgenerator/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:05generator/batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
3generator/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp<generator_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$generator/conv2d_transpose_6/BiasAddBiasAdd6generator/conv2d_transpose_6/conv2d_transpose:output:0;generator/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  ��
"generator/leaky_re_lu_12/LeakyRelu	LeakyRelu-generator/conv2d_transpose_6/BiasAdd:output:0*0
_output_shapes
:���������  ��
/generator/batch_normalization_11/ReadVariableOpReadVariableOp8generator_batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1generator/batch_normalization_11/ReadVariableOp_1ReadVariableOp:generator_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
@generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpIgenerator_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Bgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKgenerator_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
1generator/batch_normalization_11/FusedBatchNormV3FusedBatchNormV30generator/leaky_re_lu_12/LeakyRelu:activations:07generator/batch_normalization_11/ReadVariableOp:value:09generator/batch_normalization_11/ReadVariableOp_1:value:0Hgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Jgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%��'7*
is_training( �
"generator/conv2d_transpose_7/ShapeShape5generator/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:z
0generator/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2generator/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2generator/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*generator/conv2d_transpose_7/strided_sliceStridedSlice+generator/conv2d_transpose_7/Shape:output:09generator/conv2d_transpose_7/strided_slice/stack:output:0;generator/conv2d_transpose_7/strided_slice/stack_1:output:0;generator/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$generator/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@f
$generator/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@f
$generator/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
"generator/conv2d_transpose_7/stackPack3generator/conv2d_transpose_7/strided_slice:output:0-generator/conv2d_transpose_7/stack/1:output:0-generator/conv2d_transpose_7/stack/2:output:0-generator/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:|
2generator/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4generator/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4generator/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,generator/conv2d_transpose_7/strided_slice_1StridedSlice+generator/conv2d_transpose_7/stack:output:0;generator/conv2d_transpose_7/strided_slice_1/stack:output:0=generator/conv2d_transpose_7/strided_slice_1/stack_1:output:0=generator/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
<generator/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpEgenerator_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:�*
dtype0�
-generator/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput+generator/conv2d_transpose_7/stack:output:0Dgenerator/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:05generator/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
3generator/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp<generator_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$generator/conv2d_transpose_7/BiasAddBiasAdd6generator/conv2d_transpose_7/conv2d_transpose:output:0;generator/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
generator/activation_1/TanhTanh-generator/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@v
IdentityIdentitygenerator/activation_1/Tanh:y:0^NoOp*
T0*/
_output_shapes
:���������@@�

NoOpNoOpA^generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOpC^generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_10^generator/batch_normalization_10/ReadVariableOp2^generator/batch_normalization_10/ReadVariableOp_1A^generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOpC^generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_10^generator/batch_normalization_11/ReadVariableOp2^generator/batch_normalization_11/ReadVariableOp_1@^generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOpB^generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/^generator/batch_normalization_9/ReadVariableOp1^generator/batch_normalization_9/ReadVariableOp_14^generator/conv2d_transpose_4/BiasAdd/ReadVariableOp=^generator/conv2d_transpose_4/conv2d_transpose/ReadVariableOp4^generator/conv2d_transpose_5/BiasAdd/ReadVariableOp=^generator/conv2d_transpose_5/conv2d_transpose/ReadVariableOp4^generator/conv2d_transpose_6/BiasAdd/ReadVariableOp=^generator/conv2d_transpose_6/conv2d_transpose/ReadVariableOp4^generator/conv2d_transpose_7/BiasAdd/ReadVariableOp=^generator/conv2d_transpose_7/conv2d_transpose/ReadVariableOp)^generator/dense_3/BiasAdd/ReadVariableOp(^generator/dense_3/MatMul/ReadVariableOp'^generator/embedding_2/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2�
@generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp@generator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2�
Bgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Bgenerator/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12b
/generator/batch_normalization_10/ReadVariableOp/generator/batch_normalization_10/ReadVariableOp2f
1generator/batch_normalization_10/ReadVariableOp_11generator/batch_normalization_10/ReadVariableOp_12�
@generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp@generator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2�
Bgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Bgenerator/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12b
/generator/batch_normalization_11/ReadVariableOp/generator/batch_normalization_11/ReadVariableOp2f
1generator/batch_normalization_11/ReadVariableOp_11generator/batch_normalization_11/ReadVariableOp_12�
?generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?generator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Agenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Agenerator/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12`
.generator/batch_normalization_9/ReadVariableOp.generator/batch_normalization_9/ReadVariableOp2d
0generator/batch_normalization_9/ReadVariableOp_10generator/batch_normalization_9/ReadVariableOp_12j
3generator/conv2d_transpose_4/BiasAdd/ReadVariableOp3generator/conv2d_transpose_4/BiasAdd/ReadVariableOp2|
<generator/conv2d_transpose_4/conv2d_transpose/ReadVariableOp<generator/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2j
3generator/conv2d_transpose_5/BiasAdd/ReadVariableOp3generator/conv2d_transpose_5/BiasAdd/ReadVariableOp2|
<generator/conv2d_transpose_5/conv2d_transpose/ReadVariableOp<generator/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2j
3generator/conv2d_transpose_6/BiasAdd/ReadVariableOp3generator/conv2d_transpose_6/BiasAdd/ReadVariableOp2|
<generator/conv2d_transpose_6/conv2d_transpose/ReadVariableOp<generator/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2j
3generator/conv2d_transpose_7/BiasAdd/ReadVariableOp3generator/conv2d_transpose_7/BiasAdd/ReadVariableOp2|
<generator/conv2d_transpose_7/conv2d_transpose/ReadVariableOp<generator/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2T
(generator/dense_3/BiasAdd/ReadVariableOp(generator/dense_3/BiasAdd/ReadVariableOp2R
'generator/dense_3/MatMul/ReadVariableOp'generator/dense_3/MatMul/ReadVariableOp2P
&generator/embedding_2/embedding_lookup&generator/embedding_2/embedding_lookup:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
� 
�
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691044355

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042770

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
exponential_avg_factor%���=�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
6__inference_conv2d_transpose_4_layer_call_fn_691044208

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�P
�
H__inference_generator_layer_call_and_return_conditional_losses_691043656
input_5
input_6(
embedding_2_691043593:	�%
dense_3_691043598:
��  
dense_3_691043600:	� 8
conv2d_transpose_4_691043604:��+
conv2d_transpose_4_691043606:	�.
batch_normalization_9_691043610:	�.
batch_normalization_9_691043612:	�.
batch_normalization_9_691043614:	�.
batch_normalization_9_691043616:	�8
conv2d_transpose_5_691043619:��+
conv2d_transpose_5_691043621:	�/
 batch_normalization_10_691043625:	�/
 batch_normalization_10_691043627:	�/
 batch_normalization_10_691043629:	�/
 batch_normalization_10_691043631:	�8
conv2d_transpose_6_691043634:��+
conv2d_transpose_6_691043636:	�/
 batch_normalization_11_691043640:	�/
 batch_normalization_11_691043642:	�/
 batch_normalization_11_691043644:	�/
 batch_normalization_11_691043646:	�7
conv2d_transpose_7_691043649:�*
conv2d_transpose_7_691043651:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�*conv2d_transpose_6/StatefulPartitionedCall�*conv2d_transpose_7/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_6embedding_2_691043593*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_2_layer_call_and_return_conditional_losses_691043060�
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_691043070�
concatenate_2/PartitionedCallPartitionedCallinput_5"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_3_691043598dense_3_691043600*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_691043091�
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_reshape_1_layer_call_and_return_conditional_losses_691043111�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_4_691043604conv2d_transpose_4_691043606*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691042710�
leaky_re_lu_10/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691043123�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0batch_normalization_9_691043610batch_normalization_9_691043612batch_normalization_9_691043614batch_normalization_9_691043616*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691042770�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_transpose_5_691043619conv2d_transpose_5_691043621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818�
leaky_re_lu_11/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691043144�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0 batch_normalization_10_691043625 batch_normalization_10_691043627 batch_normalization_10_691043629 batch_normalization_10_691043631*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691042878�
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_transpose_6_691043634conv2d_transpose_6_691043636*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926�
leaky_re_lu_12/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0 batch_normalization_11_691043640 batch_normalization_11_691043642 batch_normalization_11_691043644 batch_normalization_11_691043646*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691042986�
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_transpose_7_691043649conv2d_transpose_7_691043651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691043034�
activation_1/PartitionedCallPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_691043186|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
i
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691043165

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:���������  �h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
� 
�
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691042818

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044295

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044523

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%��'7*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
� 
�
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691044469

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
i
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691044365

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691042926

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�9
�
"__inference__traced_save_691044686
file_prefix5
1savev2_embedding_2_embeddings_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_7_kernel_read_readvariableop6
2savev2_conv2d_transpose_7_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_2_embeddings_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop4savev2_conv2d_transpose_7_kernel_read_readvariableop2savev2_conv2d_transpose_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:
�� :� :��:�:�:�:�:�:��:�:�:�:�:�:��:�:�:�:�:�:�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:&"
 
_output_shapes
:
�� :!

_output_shapes	
:� :.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!	

_output_shapes	
:�:.
*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:�: 

_output_shapes
::

_output_shapes
: 
�
�
-__inference_generator_layer_call_fn_691043522
input_5
input_6
unknown:	�
	unknown_0:
�� 
	unknown_1:	� %
	unknown_2:��
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�%
	unknown_8:��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�&

unknown_14:��

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:�

unknown_21:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*3
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_generator_layer_call_and_return_conditional_losses_691043421w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
�
'__inference_signature_wrapper_691043710
input_5
input_6
unknown:	�
	unknown_0:
�� 
	unknown_1:	� %
	unknown_2:��
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�%
	unknown_8:��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�&

unknown_14:��

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:�

unknown_21:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__wrapped_model_691042673w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:����������:���������: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
g
K__inference_activation_1_layer_call_and_return_conditional_losses_691044593

inputs
identityN
TanhTanhinputs*
T0*/
_output_shapes
:���������@@X
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
v
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691043079

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_51
serving_default_input_5:0����������
;
input_60
serving_default_input_6:0���������H
activation_18
StatefulPartitionedCall:0���������@@tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!
embeddings"
_tf_keras_layer
"
_tf_keras_input_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

vkernel
wbias
 x_jit_compiled_convolution_op"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
!0
41
52
B3
C4
R5
S6
T7
U8
\9
]10
l11
m12
n13
o14
v15
w16
�17
�18
�19
�20
�21
�22"
trackable_list_wrapper
�
!0
41
52
B3
C4
R5
S6
\7
]8
l9
m10
v11
w12
�13
�14
�15
�16"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_generator_layer_call_fn_691043238
-__inference_generator_layer_call_fn_691043762
-__inference_generator_layer_call_fn_691043814
-__inference_generator_layer_call_fn_691043522�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_generator_layer_call_and_return_conditional_losses_691043967
H__inference_generator_layer_call_and_return_conditional_losses_691044120
H__inference_generator_layer_call_and_return_conditional_losses_691043589
H__inference_generator_layer_call_and_return_conditional_losses_691043656�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
$__inference__wrapped_model_691042673input_5input_6"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
'
!0"
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_embedding_2_layer_call_fn_691044127�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_embedding_2_layer_call_and_return_conditional_losses_691044137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'	�2embedding_2/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_flatten_3_layer_call_fn_691044142�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_flatten_3_layer_call_and_return_conditional_losses_691044148�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_concatenate_2_layer_call_fn_691044154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691044161�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_3_layer_call_fn_691044170�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_3_layer_call_and_return_conditional_losses_691044180�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
�� 2dense_3/kernel
:� 2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_reshape_1_layer_call_fn_691044185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_reshape_1_layer_call_and_return_conditional_losses_691044199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_conv2d_transpose_4_layer_call_fn_691044208�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691044241�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5:3��2conv2d_transpose_4/kernel
&:$�2conv2d_transpose_4/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_leaky_re_lu_10_layer_call_fn_691044246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691044251�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
R0
S1
T2
U3"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_9_layer_call_fn_691044264
9__inference_batch_normalization_9_layer_call_fn_691044277�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044295
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044313�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_9/gamma
):'�2batch_normalization_9/beta
2:0� (2!batch_normalization_9/moving_mean
6:4� (2%batch_normalization_9/moving_variance
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_conv2d_transpose_5_layer_call_fn_691044322�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691044355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5:3��2conv2d_transpose_5/kernel
&:$�2conv2d_transpose_5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_leaky_re_lu_11_layer_call_fn_691044360�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691044365�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_10_layer_call_fn_691044378
:__inference_batch_normalization_10_layer_call_fn_691044391�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044409
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044427�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_10/gamma
*:(�2batch_normalization_10/beta
3:1� (2"batch_normalization_10/moving_mean
7:5� (2&batch_normalization_10/moving_variance
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_conv2d_transpose_6_layer_call_fn_691044436�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691044469�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5:3��2conv2d_transpose_6/kernel
&:$�2conv2d_transpose_6/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_leaky_re_lu_12_layer_call_fn_691044474�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691044479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_11_layer_call_fn_691044492
:__inference_batch_normalization_11_layer_call_fn_691044505�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044523
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044541�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_11/gamma
*:(�2batch_normalization_11/beta
3:1� (2"batch_normalization_11/moving_mean
7:5� (2&batch_normalization_11/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_conv2d_transpose_7_layer_call_fn_691044550�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691044583�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2�2conv2d_transpose_7/kernel
%:#2conv2d_transpose_7/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_1_layer_call_fn_691044588�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_1_layer_call_and_return_conditional_losses_691044593�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
L
T0
U1
n2
o3
�4
�5"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_generator_layer_call_fn_691043238input_5input_6"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_generator_layer_call_fn_691043762inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_generator_layer_call_fn_691043814inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_generator_layer_call_fn_691043522input_5input_6"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_generator_layer_call_and_return_conditional_losses_691043967inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_generator_layer_call_and_return_conditional_losses_691044120inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_generator_layer_call_and_return_conditional_losses_691043589input_5input_6"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_generator_layer_call_and_return_conditional_losses_691043656input_5input_6"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_691043710input_5input_6"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_embedding_2_layer_call_fn_691044127inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_embedding_2_layer_call_and_return_conditional_losses_691044137inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_flatten_3_layer_call_fn_691044142inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_flatten_3_layer_call_and_return_conditional_losses_691044148inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_concatenate_2_layer_call_fn_691044154inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691044161inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_3_layer_call_fn_691044170inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_3_layer_call_and_return_conditional_losses_691044180inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_reshape_1_layer_call_fn_691044185inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_reshape_1_layer_call_and_return_conditional_losses_691044199inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_conv2d_transpose_4_layer_call_fn_691044208inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691044241inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_leaky_re_lu_10_layer_call_fn_691044246inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691044251inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_9_layer_call_fn_691044264inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_9_layer_call_fn_691044277inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044295inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044313inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_conv2d_transpose_5_layer_call_fn_691044322inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691044355inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_leaky_re_lu_11_layer_call_fn_691044360inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691044365inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_10_layer_call_fn_691044378inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_10_layer_call_fn_691044391inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044409inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044427inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_conv2d_transpose_6_layer_call_fn_691044436inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691044469inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_leaky_re_lu_12_layer_call_fn_691044474inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691044479inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_11_layer_call_fn_691044492inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_11_layer_call_fn_691044505inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044523inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044541inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_conv2d_transpose_7_layer_call_fn_691044550inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691044583inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_1_layer_call_fn_691044588inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_1_layer_call_and_return_conditional_losses_691044593inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
$__inference__wrapped_model_691042673�!45BCRSTU\]lmnovw������Y�V
O�L
J�G
"�
input_5����������
!�
input_6���������
� "C�@
>
activation_1.�+
activation_1���������@@�
K__inference_activation_1_layer_call_and_return_conditional_losses_691044593h7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
0���������@@
� �
0__inference_activation_1_layer_call_fn_691044588[7�4
-�*
(�%
inputs���������@@
� " ����������@@�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044409�lmnoN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_691044427�lmnoN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_10_layer_call_fn_691044378�lmnoN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_10_layer_call_fn_691044391�lmnoN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044523�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_691044541�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_11_layer_call_fn_691044492�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_11_layer_call_fn_691044505�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044295�RSTUN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_691044313�RSTUN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_9_layer_call_fn_691044264�RSTUN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_9_layer_call_fn_691044277�RSTUN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
L__inference_concatenate_2_layer_call_and_return_conditional_losses_691044161�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
1__inference_concatenate_2_layer_call_fn_691044154y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
Q__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_691044241�BCJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
6__inference_conv2d_transpose_4_layer_call_fn_691044208�BCJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
Q__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_691044355�\]J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
6__inference_conv2d_transpose_5_layer_call_fn_691044322�\]J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
Q__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_691044469�vwJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
6__inference_conv2d_transpose_6_layer_call_fn_691044436�vwJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
Q__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_691044583���J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������
� �
6__inference_conv2d_transpose_7_layer_call_fn_691044550���J�G
@�=
;�8
inputs,����������������������������
� "2�/+����������������������������
F__inference_dense_3_layer_call_and_return_conditional_losses_691044180^450�-
&�#
!�
inputs����������
� "&�#
�
0���������� 
� �
+__inference_dense_3_layer_call_fn_691044170Q450�-
&�#
!�
inputs����������
� "����������� �
J__inference_embedding_2_layer_call_and_return_conditional_losses_691044137`!/�,
%�"
 �
inputs���������
� "*�'
 �
0����������
� �
/__inference_embedding_2_layer_call_fn_691044127S!/�,
%�"
 �
inputs���������
� "������������
H__inference_flatten_3_layer_call_and_return_conditional_losses_691044148^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� �
-__inference_flatten_3_layer_call_fn_691044142Q4�1
*�'
%�"
inputs����������
� "������������
H__inference_generator_layer_call_and_return_conditional_losses_691043589�!45BCRSTU\]lmnovw������a�^
W�T
J�G
"�
input_5����������
!�
input_6���������
p 

 
� "-�*
#� 
0���������@@
� �
H__inference_generator_layer_call_and_return_conditional_losses_691043656�!45BCRSTU\]lmnovw������a�^
W�T
J�G
"�
input_5����������
!�
input_6���������
p

 
� "-�*
#� 
0���������@@
� �
H__inference_generator_layer_call_and_return_conditional_losses_691043967�!45BCRSTU\]lmnovw������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������
p 

 
� "-�*
#� 
0���������@@
� �
H__inference_generator_layer_call_and_return_conditional_losses_691044120�!45BCRSTU\]lmnovw������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������
p

 
� "-�*
#� 
0���������@@
� �
-__inference_generator_layer_call_fn_691043238�!45BCRSTU\]lmnovw������a�^
W�T
J�G
"�
input_5����������
!�
input_6���������
p 

 
� " ����������@@�
-__inference_generator_layer_call_fn_691043522�!45BCRSTU\]lmnovw������a�^
W�T
J�G
"�
input_5����������
!�
input_6���������
p

 
� " ����������@@�
-__inference_generator_layer_call_fn_691043762�!45BCRSTU\]lmnovw������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������
p 

 
� " ����������@@�
-__inference_generator_layer_call_fn_691043814�!45BCRSTU\]lmnovw������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������
p

 
� " ����������@@�
M__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_691044251j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
2__inference_leaky_re_lu_10_layer_call_fn_691044246]8�5
.�+
)�&
inputs����������
� "!������������
M__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_691044365j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
2__inference_leaky_re_lu_11_layer_call_fn_691044360]8�5
.�+
)�&
inputs����������
� "!������������
M__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_691044479j8�5
.�+
)�&
inputs���������  �
� ".�+
$�!
0���������  �
� �
2__inference_leaky_re_lu_12_layer_call_fn_691044474]8�5
.�+
)�&
inputs���������  �
� "!����������  ��
H__inference_reshape_1_layer_call_and_return_conditional_losses_691044199b0�-
&�#
!�
inputs���������� 
� ".�+
$�!
0����������
� �
-__inference_reshape_1_layer_call_fn_691044185U0�-
&�#
!�
inputs���������� 
� "!������������
'__inference_signature_wrapper_691043710�!45BCRSTU\]lmnovw������j�g
� 
`�]
-
input_5"�
input_5����������
,
input_6!�
input_6���������"C�@
>
activation_1.�+
activation_1���������@@