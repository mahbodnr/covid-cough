??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-0-g582c8d236cb8ʦ
?
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
:*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:*
dtype0
|
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??d* 
shared_namedense_86/kernel
u
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel* 
_output_shapes
:
??d*
dtype0
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
:d*
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

:dd*
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:d*
dtype0
?
conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_61/kernel
}
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*&
_output_shapes
:*
dtype0
t
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_61/bias
m
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes
:*
dtype0
{
dense_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_88/kernel
t
#dense_88/kernel/Read/ReadVariableOpReadVariableOpdense_88/kernel*
_output_shapes
:	?d*
dtype0
r
dense_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_88/bias
k
!dense_88/bias/Read/ReadVariableOpReadVariableOpdense_88/bias*
_output_shapes
:d*
dtype0
z
dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_89/kernel
s
#dense_89/kernel/Read/ReadVariableOpReadVariableOpdense_89/kernel*
_output_shapes

:dd*
dtype0
r
dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_89/bias
k
!dense_89/bias/Read/ReadVariableOpReadVariableOpdense_89/bias*
_output_shapes
:d*
dtype0
?
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_62/kernel
}
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*&
_output_shapes
:*
dtype0
t
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_62/bias
m
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes
:*
dtype0
z
dense_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}* 
shared_namedense_90/kernel
s
#dense_90/kernel/Read/ReadVariableOpReadVariableOpdense_90/kernel*
_output_shapes

:}*
dtype0
r
dense_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_90/bias
k
!dense_90/bias/Read/ReadVariableOpReadVariableOpdense_90/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_60/kernel/m
?
&conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_60/kernel/m*&
_output_shapes
:*
dtype0
x
conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_60/bias/m
q
$conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpconv2d_60/bias/m*
_output_shapes
:*
dtype0
?
dense_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??d*"
shared_namedense_86/kernel/m
y
%dense_86/kernel/m/Read/ReadVariableOpReadVariableOpdense_86/kernel/m* 
_output_shapes
:
??d*
dtype0
v
dense_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_86/bias/m
o
#dense_86/bias/m/Read/ReadVariableOpReadVariableOpdense_86/bias/m*
_output_shapes
:d*
dtype0
~
dense_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*"
shared_namedense_87/kernel/m
w
%dense_87/kernel/m/Read/ReadVariableOpReadVariableOpdense_87/kernel/m*
_output_shapes

:dd*
dtype0
v
dense_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_87/bias/m
o
#dense_87/bias/m/Read/ReadVariableOpReadVariableOpdense_87/bias/m*
_output_shapes
:d*
dtype0
?
conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_61/kernel/m
?
&conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_61/kernel/m*&
_output_shapes
:*
dtype0
x
conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_61/bias/m
q
$conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpconv2d_61/bias/m*
_output_shapes
:*
dtype0

dense_88/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*"
shared_namedense_88/kernel/m
x
%dense_88/kernel/m/Read/ReadVariableOpReadVariableOpdense_88/kernel/m*
_output_shapes
:	?d*
dtype0
v
dense_88/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_88/bias/m
o
#dense_88/bias/m/Read/ReadVariableOpReadVariableOpdense_88/bias/m*
_output_shapes
:d*
dtype0
~
dense_89/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*"
shared_namedense_89/kernel/m
w
%dense_89/kernel/m/Read/ReadVariableOpReadVariableOpdense_89/kernel/m*
_output_shapes

:dd*
dtype0
v
dense_89/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_89/bias/m
o
#dense_89/bias/m/Read/ReadVariableOpReadVariableOpdense_89/bias/m*
_output_shapes
:d*
dtype0
?
conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_62/kernel/m
?
&conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_62/kernel/m*&
_output_shapes
:*
dtype0
x
conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_62/bias/m
q
$conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpconv2d_62/bias/m*
_output_shapes
:*
dtype0
~
dense_90/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}*"
shared_namedense_90/kernel/m
w
%dense_90/kernel/m/Read/ReadVariableOpReadVariableOpdense_90/kernel/m*
_output_shapes

:}*
dtype0
v
dense_90/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_90/bias/m
o
#dense_90/bias/m/Read/ReadVariableOpReadVariableOpdense_90/bias/m*
_output_shapes
:*
dtype0
?
conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_60/kernel/v
?
&conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_60/kernel/v*&
_output_shapes
:*
dtype0
x
conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_60/bias/v
q
$conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpconv2d_60/bias/v*
_output_shapes
:*
dtype0
?
dense_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??d*"
shared_namedense_86/kernel/v
y
%dense_86/kernel/v/Read/ReadVariableOpReadVariableOpdense_86/kernel/v* 
_output_shapes
:
??d*
dtype0
v
dense_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_86/bias/v
o
#dense_86/bias/v/Read/ReadVariableOpReadVariableOpdense_86/bias/v*
_output_shapes
:d*
dtype0
~
dense_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*"
shared_namedense_87/kernel/v
w
%dense_87/kernel/v/Read/ReadVariableOpReadVariableOpdense_87/kernel/v*
_output_shapes

:dd*
dtype0
v
dense_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_87/bias/v
o
#dense_87/bias/v/Read/ReadVariableOpReadVariableOpdense_87/bias/v*
_output_shapes
:d*
dtype0
?
conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_61/kernel/v
?
&conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_61/kernel/v*&
_output_shapes
:*
dtype0
x
conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_61/bias/v
q
$conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpconv2d_61/bias/v*
_output_shapes
:*
dtype0

dense_88/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*"
shared_namedense_88/kernel/v
x
%dense_88/kernel/v/Read/ReadVariableOpReadVariableOpdense_88/kernel/v*
_output_shapes
:	?d*
dtype0
v
dense_88/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_88/bias/v
o
#dense_88/bias/v/Read/ReadVariableOpReadVariableOpdense_88/bias/v*
_output_shapes
:d*
dtype0
~
dense_89/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*"
shared_namedense_89/kernel/v
w
%dense_89/kernel/v/Read/ReadVariableOpReadVariableOpdense_89/kernel/v*
_output_shapes

:dd*
dtype0
v
dense_89/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namedense_89/bias/v
o
#dense_89/bias/v/Read/ReadVariableOpReadVariableOpdense_89/bias/v*
_output_shapes
:d*
dtype0
?
conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_62/kernel/v
?
&conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_62/kernel/v*&
_output_shapes
:*
dtype0
x
conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_62/bias/v
q
$conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpconv2d_62/bias/v*
_output_shapes
:*
dtype0
~
dense_90/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}*"
shared_namedense_90/kernel/v
w
%dense_90/kernel/v/Read/ReadVariableOpReadVariableOpdense_90/kernel/v*
_output_shapes

:}*
dtype0
v
dense_90/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_90/bias/v
o
#dense_90/bias/v/Read/ReadVariableOpReadVariableOpdense_90/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?g
value?gB?g B?g
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
layer-17
layer_with_weights-7
layer-18
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
R
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
h

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
R
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m?$m?%m?.m?/m?<m?=m?Fm?Gm?Pm?Qm?^m?_m?lm?mm?v?v?$v?%v?.v?/v?<v?=v?Fv?Gv?Pv?Qv?^v?_v?lv?mv?
 
v
0
1
$2
%3
.4
/5
<6
=7
F8
G9
P10
Q11
^12
_13
l14
m15
v
0
1
$2
%3
.4
/5
<6
=7
F8
G9
P10
Q11
^12
_13
l14
m15
?
regularization_losses
wlayer_metrics
xlayer_regularization_losses

ylayers
zmetrics
{non_trainable_variables
trainable_variables
	variables
 
\Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
|layer_metrics
}layer_regularization_losses

~layers
metrics
?non_trainable_variables
trainable_variables
	variables
 
 
 
?
 regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
!trainable_variables
"	variables
[Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_86/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
?
&regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
'trainable_variables
(	variables
 
 
 
?
*regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
+trainable_variables
,	variables
[Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_87/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
0regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
1trainable_variables
2	variables
 
 
 
?
4regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
5trainable_variables
6	variables
 
 
 
?
8regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
9trainable_variables
:	variables
\Z
VARIABLE_VALUEconv2d_61/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_61/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
?
>regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
?trainable_variables
@	variables
 
 
 
?
Bregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Ctrainable_variables
D	variables
[Y
VARIABLE_VALUEdense_88/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_88/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
Hregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Itrainable_variables
J	variables
 
 
 
?
Lregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Mtrainable_variables
N	variables
[Y
VARIABLE_VALUEdense_89/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_89/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
Rregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Strainable_variables
T	variables
 
 
 
?
Vregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Wtrainable_variables
X	variables
 
 
 
?
Zregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
[trainable_variables
\	variables
\Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
?
`regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
atrainable_variables
b	variables
 
 
 
?
dregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
etrainable_variables
f	variables
 
 
 
?
hregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
itrainable_variables
j	variables
[Y
VARIABLE_VALUEdense_90/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_90/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
?
nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
otrainable_variables
p	variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
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
17
18

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
zx
VARIABLE_VALUEconv2d_60/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_60/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_86/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_86/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_87/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_87/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_61/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_61/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_88/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_88/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_89/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_89/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_62/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_62/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_90/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_90/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_60/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_60/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_86/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_86/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_87/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_87/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_61/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_61/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_88/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_88/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_89/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_89/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_62/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_62/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_90/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_90/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_24Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_24conv2d_60/kernelconv2d_60/biasdense_86/kerneldense_86/biasdense_87/kerneldense_87/biasconv2d_61/kernelconv2d_61/biasdense_88/kerneldense_88/biasdense_89/kerneldense_89/biasconv2d_62/kernelconv2d_62/biasdense_90/kerneldense_90/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_1511
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp#dense_88/kernel/Read/ReadVariableOp!dense_88/bias/Read/ReadVariableOp#dense_89/kernel/Read/ReadVariableOp!dense_89/bias/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp#dense_90/kernel/Read/ReadVariableOp!dense_90/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&conv2d_60/kernel/m/Read/ReadVariableOp$conv2d_60/bias/m/Read/ReadVariableOp%dense_86/kernel/m/Read/ReadVariableOp#dense_86/bias/m/Read/ReadVariableOp%dense_87/kernel/m/Read/ReadVariableOp#dense_87/bias/m/Read/ReadVariableOp&conv2d_61/kernel/m/Read/ReadVariableOp$conv2d_61/bias/m/Read/ReadVariableOp%dense_88/kernel/m/Read/ReadVariableOp#dense_88/bias/m/Read/ReadVariableOp%dense_89/kernel/m/Read/ReadVariableOp#dense_89/bias/m/Read/ReadVariableOp&conv2d_62/kernel/m/Read/ReadVariableOp$conv2d_62/bias/m/Read/ReadVariableOp%dense_90/kernel/m/Read/ReadVariableOp#dense_90/bias/m/Read/ReadVariableOp&conv2d_60/kernel/v/Read/ReadVariableOp$conv2d_60/bias/v/Read/ReadVariableOp%dense_86/kernel/v/Read/ReadVariableOp#dense_86/bias/v/Read/ReadVariableOp%dense_87/kernel/v/Read/ReadVariableOp#dense_87/bias/v/Read/ReadVariableOp&conv2d_61/kernel/v/Read/ReadVariableOp$conv2d_61/bias/v/Read/ReadVariableOp%dense_88/kernel/v/Read/ReadVariableOp#dense_88/bias/v/Read/ReadVariableOp%dense_89/kernel/v/Read/ReadVariableOp#dense_89/bias/v/Read/ReadVariableOp&conv2d_62/kernel/v/Read/ReadVariableOp$conv2d_62/bias/v/Read/ReadVariableOp%dense_90/kernel/v/Read/ReadVariableOp#dense_90/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2459
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_60/kernelconv2d_60/biasdense_86/kerneldense_86/biasdense_87/kerneldense_87/biasconv2d_61/kernelconv2d_61/biasdense_88/kerneldense_88/biasdense_89/kerneldense_89/biasconv2d_62/kernelconv2d_62/biasdense_90/kerneldense_90/biasiterbeta_1beta_2decaylearning_ratetotalcounttotal_1count_1conv2d_60/kernel/mconv2d_60/bias/mdense_86/kernel/mdense_86/bias/mdense_87/kernel/mdense_87/bias/mconv2d_61/kernel/mconv2d_61/bias/mdense_88/kernel/mdense_88/bias/mdense_89/kernel/mdense_89/bias/mconv2d_62/kernel/mconv2d_62/bias/mdense_90/kernel/mdense_90/bias/mconv2d_60/kernel/vconv2d_60/bias/vdense_86/kernel/vdense_86/bias/vdense_87/kernel/vdense_87/bias/vconv2d_61/kernel/vconv2d_61/bias/vdense_88/kernel/vdense_88/bias/vdense_89/kernel/vdense_89/bias/vconv2d_62/kernel/vconv2d_62/bias/vdense_90/kernel/vdense_90/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2640??
?	
?
B__inference_dense_90_layer_call_and_return_conditional_losses_1090

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????}
 
_user_specified_nameinputs
?u
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1131
input_24
conv2d_60_683
conv2d_60_685
dense_86_730
dense_86_732
dense_87_793
dense_87_795
conv2d_61_871
conv2d_61_873
dense_88_918
dense_88_920
dense_89_981
dense_89_983
conv2d_62_1059
conv2d_62_1061
dense_90_1101
dense_90_1103
identity??!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall? dense_86/StatefulPartitionedCall?1dense_86/kernel/Regularizer/Square/ReadVariableOp? dense_87/StatefulPartitionedCall?1dense_87/kernel/Regularizer/Square/ReadVariableOp? dense_88/StatefulPartitionedCall?1dense_88/kernel/Regularizer/Square/ReadVariableOp? dense_89/StatefulPartitionedCall?1dense_89/kernel/Regularizer/Square/ReadVariableOp? dense_90/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinput_24conv2d_60_683conv2d_60_685*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_60_layer_call_and_return_conditional_losses_6722#
!conv2d_60/StatefulPartitionedCall?
flatten_44/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_44_layer_call_and_return_conditional_losses_6942
flatten_44/PartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCall#flatten_44/PartitionedCall:output:0dense_86_730dense_86_732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_86_layer_call_and_return_conditional_losses_7192"
 dense_86/StatefulPartitionedCall?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7472$
"dropout_56/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall+dropout_56/StatefulPartitionedCall:output:0dense_87_793dense_87_795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_87_layer_call_and_return_conditional_losses_7822"
 dense_87/StatefulPartitionedCall?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8102$
"dropout_57/StatefulPartitionedCall?
reshape_22/PartitionedCallPartitionedCall+dropout_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_22_layer_call_and_return_conditional_losses_8422
reshape_22/PartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall#reshape_22/PartitionedCall:output:0conv2d_61_871conv2d_61_873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_61_layer_call_and_return_conditional_losses_8602#
!conv2d_61/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_45_layer_call_and_return_conditional_losses_8822
flatten_45/PartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCall#flatten_45/PartitionedCall:output:0dense_88_918dense_88_920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_88_layer_call_and_return_conditional_losses_9072"
 dense_88/StatefulPartitionedCall?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0#^dropout_57/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9352$
"dropout_58/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0dense_89_981dense_89_983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_89_layer_call_and_return_conditional_losses_9702"
 dense_89/StatefulPartitionedCall?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_59_layer_call_and_return_conditional_losses_9982$
"dropout_59/StatefulPartitionedCall?
reshape_23/PartitionedCallPartitionedCall+dropout_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_23_layer_call_and_return_conditional_losses_10302
reshape_23/PartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall#reshape_23/PartitionedCall:output:0conv2d_62_1059conv2d_62_1061*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_62_layer_call_and_return_conditional_losses_10482#
!conv2d_62/StatefulPartitionedCall?
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_6522&
$average_pooling2d_10/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????}* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_46_layer_call_and_return_conditional_losses_10712
flatten_46/PartitionedCall?
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_46/PartitionedCall:output:0dense_90_1101dense_90_1103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_90_layer_call_and_return_conditional_losses_10902"
 dense_90/StatefulPartitionedCall?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_86_730* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_793*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_88_918*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_89_981*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall2^dense_86/kernel/Regularizer/Square/ReadVariableOp!^dense_87/StatefulPartitionedCall2^dense_87/kernel/Regularizer/Square/ReadVariableOp!^dense_88/StatefulPartitionedCall2^dense_88/kernel/Regularizer/Square/ReadVariableOp!^dense_89/StatefulPartitionedCall2^dense_89/kernel/Regularizer/Square/ReadVariableOp!^dense_90/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?
|
'__inference_dense_87_layer_call_fn_1958

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_87_layer_call_and_return_conditional_losses_7822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
`
D__inference_flatten_46_layer_call_and_return_conditional_losses_2196

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????}   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????}2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_45_layer_call_and_return_conditional_losses_882

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?

?
,__inference_functional_45_layer_call_fn_1325
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_45_layer_call_and_return_conditional_losses_12902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?

?
,__inference_functional_45_layer_call_fn_1837

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_45_layer_call_and_return_conditional_losses_14052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_58_layer_call_fn_2088

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
B__inference_conv2d_61_layer_call_and_return_conditional_losses_860

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
b
)__inference_dropout_59_layer_call_fn_2147

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_59_layer_call_and_return_conditional_losses_9982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
|
'__inference_dense_88_layer_call_fn_2066

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_88_layer_call_and_return_conditional_losses_9072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_22_layer_call_and_return_conditional_losses_1999

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????

2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
}
(__inference_conv2d_60_layer_call_fn_1856

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_60_layer_call_and_return_conditional_losses_6722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_62_layer_call_fn_2190

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_62_layer_call_and_return_conditional_losses_10482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?	
?
C__inference_conv2d_60_layer_call_and_return_conditional_losses_1847

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_2640
file_prefix%
!assignvariableop_conv2d_60_kernel%
!assignvariableop_1_conv2d_60_bias&
"assignvariableop_2_dense_86_kernel$
 assignvariableop_3_dense_86_bias&
"assignvariableop_4_dense_87_kernel$
 assignvariableop_5_dense_87_bias'
#assignvariableop_6_conv2d_61_kernel%
!assignvariableop_7_conv2d_61_bias&
"assignvariableop_8_dense_88_kernel$
 assignvariableop_9_dense_88_bias'
#assignvariableop_10_dense_89_kernel%
!assignvariableop_11_dense_89_bias(
$assignvariableop_12_conv2d_62_kernel&
"assignvariableop_13_conv2d_62_bias'
#assignvariableop_14_dense_90_kernel%
!assignvariableop_15_dense_90_bias
assignvariableop_16_iter
assignvariableop_17_beta_1
assignvariableop_18_beta_2
assignvariableop_19_decay%
!assignvariableop_20_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1*
&assignvariableop_25_conv2d_60_kernel_m(
$assignvariableop_26_conv2d_60_bias_m)
%assignvariableop_27_dense_86_kernel_m'
#assignvariableop_28_dense_86_bias_m)
%assignvariableop_29_dense_87_kernel_m'
#assignvariableop_30_dense_87_bias_m*
&assignvariableop_31_conv2d_61_kernel_m(
$assignvariableop_32_conv2d_61_bias_m)
%assignvariableop_33_dense_88_kernel_m'
#assignvariableop_34_dense_88_bias_m)
%assignvariableop_35_dense_89_kernel_m'
#assignvariableop_36_dense_89_bias_m*
&assignvariableop_37_conv2d_62_kernel_m(
$assignvariableop_38_conv2d_62_bias_m)
%assignvariableop_39_dense_90_kernel_m'
#assignvariableop_40_dense_90_bias_m*
&assignvariableop_41_conv2d_60_kernel_v(
$assignvariableop_42_conv2d_60_bias_v)
%assignvariableop_43_dense_86_kernel_v'
#assignvariableop_44_dense_86_bias_v)
%assignvariableop_45_dense_87_kernel_v'
#assignvariableop_46_dense_87_bias_v*
&assignvariableop_47_conv2d_61_kernel_v(
$assignvariableop_48_conv2d_61_bias_v)
%assignvariableop_49_dense_88_kernel_v'
#assignvariableop_50_dense_88_bias_v)
%assignvariableop_51_dense_89_kernel_v'
#assignvariableop_52_dense_89_bias_v*
&assignvariableop_53_conv2d_62_kernel_v(
$assignvariableop_54_conv2d_62_bias_v)
%assignvariableop_55_dense_90_kernel_v'
#assignvariableop_56_dense_90_bias_v
identity_58??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_60_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_60_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_86_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_86_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_87_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_87_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_61_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_61_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_88_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_88_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_89_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_89_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_62_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_62_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_90_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_90_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp&assignvariableop_25_conv2d_60_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_60_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_dense_86_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_86_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_dense_87_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_87_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_conv2d_61_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_61_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_dense_88_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_88_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_dense_89_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_89_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp&assignvariableop_37_conv2d_62_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_62_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp%assignvariableop_39_dense_90_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_90_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp&assignvariableop_41_conv2d_60_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_60_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp%assignvariableop_43_dense_86_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp#assignvariableop_44_dense_86_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp%assignvariableop_45_dense_87_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp#assignvariableop_46_dense_87_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp&assignvariableop_47_conv2d_61_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_61_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp%assignvariableop_49_dense_88_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp#assignvariableop_50_dense_88_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_dense_89_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp#assignvariableop_52_dense_89_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp&assignvariableop_53_conv2d_62_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp$assignvariableop_54_conv2d_62_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp%assignvariableop_55_dense_90_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp#assignvariableop_56_dense_90_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57?

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1651

inputs,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource,
(conv2d_61_conv2d_readvariableop_resource-
)conv2d_61_biasadd_readvariableop_resource+
'dense_88_matmul_readvariableop_resource,
(dense_88_biasadd_readvariableop_resource+
'dense_89_matmul_readvariableop_resource,
(dense_89_biasadd_readvariableop_resource,
(conv2d_62_conv2d_readvariableop_resource-
)conv2d_62_biasadd_readvariableop_resource+
'dense_90_matmul_readvariableop_resource,
(dense_90_biasadd_readvariableop_resource
identity?? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp?dense_86/BiasAdd/ReadVariableOp?dense_86/MatMul/ReadVariableOp?1dense_86/kernel/Regularizer/Square/ReadVariableOp?dense_87/BiasAdd/ReadVariableOp?dense_87/MatMul/ReadVariableOp?1dense_87/kernel/Regularizer/Square/ReadVariableOp?dense_88/BiasAdd/ReadVariableOp?dense_88/MatMul/ReadVariableOp?1dense_88/kernel/Regularizer/Square/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?1dense_89/kernel/Regularizer/Square/ReadVariableOp?dense_90/BiasAdd/ReadVariableOp?dense_90/MatMul/ReadVariableOp?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_60/BiasAddu
flatten_44/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_44/Const?
flatten_44/ReshapeReshapeconv2d_60/BiasAdd:output:0flatten_44/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_44/Reshape?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulflatten_44/Reshape:output:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_86/MatMul?
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_86/BiasAdd/ReadVariableOp?
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_86/Reluy
dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_56/dropout/Const?
dropout_56/dropout/MulMuldense_86/Relu:activations:0!dropout_56/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_56/dropout/Mul
dropout_56/dropout/ShapeShapedense_86/Relu:activations:0*
T0*
_output_shapes
:2
dropout_56/dropout/Shape?
/dropout_56/dropout/random_uniform/RandomUniformRandomUniform!dropout_56/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_56/dropout/random_uniform/RandomUniform?
!dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_56/dropout/GreaterEqual/y?
dropout_56/dropout/GreaterEqualGreaterEqual8dropout_56/dropout/random_uniform/RandomUniform:output:0*dropout_56/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_56/dropout/GreaterEqual?
dropout_56/dropout/CastCast#dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_56/dropout/Cast?
dropout_56/dropout/Mul_1Muldropout_56/dropout/Mul:z:0dropout_56/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_56/dropout/Mul_1?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMuldropout_56/dropout/Mul_1:z:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_87/MatMul?
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_87/BiasAdd/ReadVariableOp?
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_87/BiasAdds
dense_87/ReluReludense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_87/Reluy
dropout_57/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_57/dropout/Const?
dropout_57/dropout/MulMuldense_87/Relu:activations:0!dropout_57/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_57/dropout/Mul
dropout_57/dropout/ShapeShapedense_87/Relu:activations:0*
T0*
_output_shapes
:2
dropout_57/dropout/Shape?
/dropout_57/dropout/random_uniform/RandomUniformRandomUniform!dropout_57/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_57/dropout/random_uniform/RandomUniform?
!dropout_57/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_57/dropout/GreaterEqual/y?
dropout_57/dropout/GreaterEqualGreaterEqual8dropout_57/dropout/random_uniform/RandomUniform:output:0*dropout_57/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_57/dropout/GreaterEqual?
dropout_57/dropout/CastCast#dropout_57/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_57/dropout/Cast?
dropout_57/dropout/Mul_1Muldropout_57/dropout/Mul:z:0dropout_57/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_57/dropout/Mul_1p
reshape_22/ShapeShapedropout_57/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
reshape_22/Shape?
reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_22/strided_slice/stack?
 reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_22/strided_slice/stack_1?
 reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_22/strided_slice/stack_2?
reshape_22/strided_sliceStridedSlicereshape_22/Shape:output:0'reshape_22/strided_slice/stack:output:0)reshape_22/strided_slice/stack_1:output:0)reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_22/strided_slicez
reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_22/Reshape/shape/1z
reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_22/Reshape/shape/2z
reshape_22/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_22/Reshape/shape/3?
reshape_22/Reshape/shapePack!reshape_22/strided_slice:output:0#reshape_22/Reshape/shape/1:output:0#reshape_22/Reshape/shape/2:output:0#reshape_22/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_22/Reshape/shape?
reshape_22/ReshapeReshapedropout_57/dropout/Mul_1:z:0!reshape_22/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2
reshape_22/Reshape?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_61/Conv2D/ReadVariableOp?
conv2d_61/Conv2DConv2Dreshape_22/Reshape:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
conv2d_61/Conv2D?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
conv2d_61/BiasAddu
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_45/Const?
flatten_45/ReshapeReshapeconv2d_61/BiasAdd:output:0flatten_45/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_45/Reshape?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulflatten_45/Reshape:output:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_88/MatMul?
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_88/BiasAdd/ReadVariableOp?
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_88/BiasAdds
dense_88/ReluReludense_88/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_88/Reluy
dropout_58/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_58/dropout/Const?
dropout_58/dropout/MulMuldense_88/Relu:activations:0!dropout_58/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_58/dropout/Mul
dropout_58/dropout/ShapeShapedense_88/Relu:activations:0*
T0*
_output_shapes
:2
dropout_58/dropout/Shape?
/dropout_58/dropout/random_uniform/RandomUniformRandomUniform!dropout_58/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_58/dropout/random_uniform/RandomUniform?
!dropout_58/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_58/dropout/GreaterEqual/y?
dropout_58/dropout/GreaterEqualGreaterEqual8dropout_58/dropout/random_uniform/RandomUniform:output:0*dropout_58/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_58/dropout/GreaterEqual?
dropout_58/dropout/CastCast#dropout_58/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_58/dropout/Cast?
dropout_58/dropout/Mul_1Muldropout_58/dropout/Mul:z:0dropout_58/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_58/dropout/Mul_1?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMuldropout_58/dropout/Mul_1:z:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_89/BiasAdds
dense_89/ReluReludense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_89/Reluy
dropout_59/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_59/dropout/Const?
dropout_59/dropout/MulMuldense_89/Relu:activations:0!dropout_59/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_59/dropout/Mul
dropout_59/dropout/ShapeShapedense_89/Relu:activations:0*
T0*
_output_shapes
:2
dropout_59/dropout/Shape?
/dropout_59/dropout/random_uniform/RandomUniformRandomUniform!dropout_59/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_59/dropout/random_uniform/RandomUniform?
!dropout_59/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_59/dropout/GreaterEqual/y?
dropout_59/dropout/GreaterEqualGreaterEqual8dropout_59/dropout/random_uniform/RandomUniform:output:0*dropout_59/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_59/dropout/GreaterEqual?
dropout_59/dropout/CastCast#dropout_59/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_59/dropout/Cast?
dropout_59/dropout/Mul_1Muldropout_59/dropout/Mul:z:0dropout_59/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_59/dropout/Mul_1p
reshape_23/ShapeShapedropout_59/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
reshape_23/Shape?
reshape_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_23/strided_slice/stack?
 reshape_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_23/strided_slice/stack_1?
 reshape_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_23/strided_slice/stack_2?
reshape_23/strided_sliceStridedSlicereshape_23/Shape:output:0'reshape_23/strided_slice/stack:output:0)reshape_23/strided_slice/stack_1:output:0)reshape_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_23/strided_slicez
reshape_23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_23/Reshape/shape/1z
reshape_23/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_23/Reshape/shape/2z
reshape_23/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_23/Reshape/shape/3?
reshape_23/Reshape/shapePack!reshape_23/strided_slice:output:0#reshape_23/Reshape/shape/1:output:0#reshape_23/Reshape/shape/2:output:0#reshape_23/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_23/Reshape/shape?
reshape_23/ReshapeReshapedropout_59/dropout/Mul_1:z:0!reshape_23/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2
reshape_23/Reshape?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2Dreshape_23/Reshape:output:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
conv2d_62/BiasAdd?
average_pooling2d_10/AvgPoolAvgPoolconv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
average_pooling2d_10/AvgPoolu
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????}   2
flatten_46/Const?
flatten_46/ReshapeReshape%average_pooling2d_10/AvgPool:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????}2
flatten_46/Reshape?
dense_90/MatMul/ReadVariableOpReadVariableOp'dense_90_matmul_readvariableop_resource*
_output_shapes

:}*
dtype02 
dense_90/MatMul/ReadVariableOp?
dense_90/MatMulMatMulflatten_46/Reshape:output:0&dense_90/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_90/MatMul?
dense_90/BiasAdd/ReadVariableOpReadVariableOp(dense_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_90/BiasAdd/ReadVariableOp?
dense_90/BiasAddBiasAdddense_90/MatMul:product:0'dense_90/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_90/BiasAdd|
dense_90/SoftmaxSoftmaxdense_90/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_90/Softmax?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentitydense_90/Softmax:softmax:0!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp2^dense_86/kernel/Regularizer/Square/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp2^dense_87/kernel/Regularizer/Square/ReadVariableOp ^dense_88/BiasAdd/ReadVariableOp^dense_88/MatMul/ReadVariableOp2^dense_88/kernel/Regularizer/Square/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp2^dense_89/kernel/Regularizer/Square/ReadVariableOp ^dense_90/BiasAdd/ReadVariableOp^dense_90/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2B
dense_88/BiasAdd/ReadVariableOpdense_88/BiasAdd/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2B
dense_90/BiasAdd/ReadVariableOpdense_90/BiasAdd/ReadVariableOp2@
dense_90/MatMul/ReadVariableOpdense_90/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_reshape_22_layer_call_and_return_conditional_losses_842

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????

2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
b
D__inference_dropout_58_layer_call_and_return_conditional_losses_2083

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_dense_87_layer_call_and_return_conditional_losses_1949

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_87/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_87/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_dropout_58_layer_call_fn_2093

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
|
'__inference_dense_86_layer_call_fn_1899

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_86_layer_call_and_return_conditional_losses_7192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_dense_88_layer_call_and_return_conditional_losses_2057

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_88/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_88/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_44_layer_call_and_return_conditional_losses_1862

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_90_layer_call_and_return_conditional_losses_2212

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????}
 
_user_specified_nameinputs
?

b
C__inference_dropout_57_layer_call_and_return_conditional_losses_810

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
C__inference_dropout_57_layer_call_and_return_conditional_losses_815

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

b
C__inference_dropout_56_layer_call_and_return_conditional_losses_747

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
,__inference_functional_45_layer_call_fn_1800

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_45_layer_call_and_return_conditional_losses_12902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_61_layer_call_fn_2023

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_61_layer_call_and_return_conditional_losses_8602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
E
)__inference_flatten_45_layer_call_fn_2034

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_45_layer_call_and_return_conditional_losses_8822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
`
D__inference_flatten_45_layer_call_and_return_conditional_losses_2029

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
|
'__inference_dense_89_layer_call_fn_2125

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_89_layer_call_and_return_conditional_losses_9702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_reshape_23_layer_call_fn_2171

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_23_layer_call_and_return_conditional_losses_10302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_62_layer_call_and_return_conditional_losses_1048

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
`
D__inference_reshape_23_layer_call_and_return_conditional_losses_1030

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????

2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
A__inference_dense_87_layer_call_and_return_conditional_losses_782

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_87/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_87/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?o
?
__inference__traced_save_2459
file_prefix/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop.
*savev2_dense_88_kernel_read_readvariableop,
(savev2_dense_88_bias_read_readvariableop.
*savev2_dense_89_kernel_read_readvariableop,
(savev2_dense_89_bias_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop.
*savev2_dense_90_kernel_read_readvariableop,
(savev2_dense_90_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop1
-savev2_conv2d_60_kernel_m_read_readvariableop/
+savev2_conv2d_60_bias_m_read_readvariableop0
,savev2_dense_86_kernel_m_read_readvariableop.
*savev2_dense_86_bias_m_read_readvariableop0
,savev2_dense_87_kernel_m_read_readvariableop.
*savev2_dense_87_bias_m_read_readvariableop1
-savev2_conv2d_61_kernel_m_read_readvariableop/
+savev2_conv2d_61_bias_m_read_readvariableop0
,savev2_dense_88_kernel_m_read_readvariableop.
*savev2_dense_88_bias_m_read_readvariableop0
,savev2_dense_89_kernel_m_read_readvariableop.
*savev2_dense_89_bias_m_read_readvariableop1
-savev2_conv2d_62_kernel_m_read_readvariableop/
+savev2_conv2d_62_bias_m_read_readvariableop0
,savev2_dense_90_kernel_m_read_readvariableop.
*savev2_dense_90_bias_m_read_readvariableop1
-savev2_conv2d_60_kernel_v_read_readvariableop/
+savev2_conv2d_60_bias_v_read_readvariableop0
,savev2_dense_86_kernel_v_read_readvariableop.
*savev2_dense_86_bias_v_read_readvariableop0
,savev2_dense_87_kernel_v_read_readvariableop.
*savev2_dense_87_bias_v_read_readvariableop1
-savev2_conv2d_61_kernel_v_read_readvariableop/
+savev2_conv2d_61_bias_v_read_readvariableop0
,savev2_dense_88_kernel_v_read_readvariableop.
*savev2_dense_88_bias_v_read_readvariableop0
,savev2_dense_89_kernel_v_read_readvariableop.
*savev2_dense_89_bias_v_read_readvariableop1
-savev2_conv2d_62_kernel_v_read_readvariableop/
+savev2_conv2d_62_bias_v_read_readvariableop0
,savev2_dense_90_kernel_v_read_readvariableop.
*savev2_dense_90_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop*savev2_dense_88_kernel_read_readvariableop(savev2_dense_88_bias_read_readvariableop*savev2_dense_89_kernel_read_readvariableop(savev2_dense_89_bias_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop*savev2_dense_90_kernel_read_readvariableop(savev2_dense_90_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_conv2d_60_kernel_m_read_readvariableop+savev2_conv2d_60_bias_m_read_readvariableop,savev2_dense_86_kernel_m_read_readvariableop*savev2_dense_86_bias_m_read_readvariableop,savev2_dense_87_kernel_m_read_readvariableop*savev2_dense_87_bias_m_read_readvariableop-savev2_conv2d_61_kernel_m_read_readvariableop+savev2_conv2d_61_bias_m_read_readvariableop,savev2_dense_88_kernel_m_read_readvariableop*savev2_dense_88_bias_m_read_readvariableop,savev2_dense_89_kernel_m_read_readvariableop*savev2_dense_89_bias_m_read_readvariableop-savev2_conv2d_62_kernel_m_read_readvariableop+savev2_conv2d_62_bias_m_read_readvariableop,savev2_dense_90_kernel_m_read_readvariableop*savev2_dense_90_bias_m_read_readvariableop-savev2_conv2d_60_kernel_v_read_readvariableop+savev2_conv2d_60_bias_v_read_readvariableop,savev2_dense_86_kernel_v_read_readvariableop*savev2_dense_86_bias_v_read_readvariableop,savev2_dense_87_kernel_v_read_readvariableop*savev2_dense_87_bias_v_read_readvariableop-savev2_conv2d_61_kernel_v_read_readvariableop+savev2_conv2d_61_bias_v_read_readvariableop,savev2_dense_88_kernel_v_read_readvariableop*savev2_dense_88_bias_v_read_readvariableop,savev2_dense_89_kernel_v_read_readvariableop*savev2_dense_89_bias_v_read_readvariableop-savev2_conv2d_62_kernel_v_read_readvariableop+savev2_conv2d_62_bias_v_read_readvariableop,savev2_dense_90_kernel_v_read_readvariableop*savev2_dense_90_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::
??d:d:dd:d:::	?d:d:dd:d:::}:: : : : : : : : : :::
??d:d:dd:d:::	?d:d:dd:d:::}::::
??d:d:dd:d:::	?d:d:dd:d:::}:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:,(
&
_output_shapes
:: 

_output_shapes
::%	!

_output_shapes
:	?d: 


_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:}: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:, (
&
_output_shapes
:: !

_output_shapes
::%"!

_output_shapes
:	?d: #

_output_shapes
:d:$$ 

_output_shapes

:dd: %

_output_shapes
:d:,&(
&
_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:}: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::&,"
 
_output_shapes
:
??d: -

_output_shapes
:d:$. 

_output_shapes

:dd: /

_output_shapes
:d:,0(
&
_output_shapes
:: 1

_output_shapes
::%2!

_output_shapes
:	?d: 3

_output_shapes
:d:$4 

_output_shapes

:dd: 5

_output_shapes
:d:,6(
&
_output_shapes
:: 7

_output_shapes
::$8 

_output_shapes

:}: 9

_output_shapes
:::

_output_shapes
: 
?
?
__inference_loss_fn_2_2254>
:dense_88_kernel_regularizer_square_readvariableop_resource
identity??1dense_88/kernel/Regularizer/Square/ReadVariableOp?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_88_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
IdentityIdentity#dense_88/kernel/Regularizer/mul:z:02^dense_88/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp
?	
?
C__inference_conv2d_62_layer_call_and_return_conditional_losses_2181

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
E
)__inference_dropout_56_layer_call_fn_1926

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
B__inference_conv2d_60_layer_call_and_return_conditional_losses_672

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?u
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1290

inputs
conv2d_60_1215
conv2d_60_1217
dense_86_1221
dense_86_1223
dense_87_1227
dense_87_1229
conv2d_61_1234
conv2d_61_1236
dense_88_1240
dense_88_1242
dense_89_1246
dense_89_1248
conv2d_62_1253
conv2d_62_1255
dense_90_1260
dense_90_1262
identity??!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall? dense_86/StatefulPartitionedCall?1dense_86/kernel/Regularizer/Square/ReadVariableOp? dense_87/StatefulPartitionedCall?1dense_87/kernel/Regularizer/Square/ReadVariableOp? dense_88/StatefulPartitionedCall?1dense_88/kernel/Regularizer/Square/ReadVariableOp? dense_89/StatefulPartitionedCall?1dense_89/kernel/Regularizer/Square/ReadVariableOp? dense_90/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_1215conv2d_60_1217*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_60_layer_call_and_return_conditional_losses_6722#
!conv2d_60/StatefulPartitionedCall?
flatten_44/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_44_layer_call_and_return_conditional_losses_6942
flatten_44/PartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCall#flatten_44/PartitionedCall:output:0dense_86_1221dense_86_1223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_86_layer_call_and_return_conditional_losses_7192"
 dense_86/StatefulPartitionedCall?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7472$
"dropout_56/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall+dropout_56/StatefulPartitionedCall:output:0dense_87_1227dense_87_1229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_87_layer_call_and_return_conditional_losses_7822"
 dense_87/StatefulPartitionedCall?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8102$
"dropout_57/StatefulPartitionedCall?
reshape_22/PartitionedCallPartitionedCall+dropout_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_22_layer_call_and_return_conditional_losses_8422
reshape_22/PartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall#reshape_22/PartitionedCall:output:0conv2d_61_1234conv2d_61_1236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_61_layer_call_and_return_conditional_losses_8602#
!conv2d_61/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_45_layer_call_and_return_conditional_losses_8822
flatten_45/PartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCall#flatten_45/PartitionedCall:output:0dense_88_1240dense_88_1242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_88_layer_call_and_return_conditional_losses_9072"
 dense_88/StatefulPartitionedCall?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0#^dropout_57/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9352$
"dropout_58/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0dense_89_1246dense_89_1248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_89_layer_call_and_return_conditional_losses_9702"
 dense_89/StatefulPartitionedCall?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_59_layer_call_and_return_conditional_losses_9982$
"dropout_59/StatefulPartitionedCall?
reshape_23/PartitionedCallPartitionedCall+dropout_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_23_layer_call_and_return_conditional_losses_10302
reshape_23/PartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall#reshape_23/PartitionedCall:output:0conv2d_62_1253conv2d_62_1255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_62_layer_call_and_return_conditional_losses_10482#
!conv2d_62/StatefulPartitionedCall?
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_6522&
$average_pooling2d_10/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????}* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_46_layer_call_and_return_conditional_losses_10712
flatten_46/PartitionedCall?
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_46/PartitionedCall:output:0dense_90_1260dense_90_1262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_90_layer_call_and_return_conditional_losses_10902"
 dense_90/StatefulPartitionedCall?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_86_1221* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_1227*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_88_1240*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_89_1246*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall2^dense_86/kernel/Regularizer/Square/ReadVariableOp!^dense_87/StatefulPartitionedCall2^dense_87/kernel/Regularizer/Square/ReadVariableOp!^dense_88/StatefulPartitionedCall2^dense_88/kernel/Regularizer/Square/ReadVariableOp!^dense_89/StatefulPartitionedCall2^dense_89/kernel/Regularizer/Square/ReadVariableOp!^dense_90/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_dense_88_layer_call_and_return_conditional_losses_907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_88/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_88/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

b
C__inference_dropout_58_layer_call_and_return_conditional_losses_935

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
b
D__inference_dropout_59_layer_call_and_return_conditional_losses_2142

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_dropout_57_layer_call_fn_1985

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_2243>
:dense_87_kernel_regularizer_square_readvariableop_resource
identity??1dense_87/kernel/Regularizer/Square/ReadVariableOp?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_87_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
IdentityIdentity#dense_87/kernel/Regularizer/mul:z:02^dense_87/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp
?
c
D__inference_dropout_57_layer_call_and_return_conditional_losses_1970

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_646
input_24:
6functional_45_conv2d_60_conv2d_readvariableop_resource;
7functional_45_conv2d_60_biasadd_readvariableop_resource9
5functional_45_dense_86_matmul_readvariableop_resource:
6functional_45_dense_86_biasadd_readvariableop_resource9
5functional_45_dense_87_matmul_readvariableop_resource:
6functional_45_dense_87_biasadd_readvariableop_resource:
6functional_45_conv2d_61_conv2d_readvariableop_resource;
7functional_45_conv2d_61_biasadd_readvariableop_resource9
5functional_45_dense_88_matmul_readvariableop_resource:
6functional_45_dense_88_biasadd_readvariableop_resource9
5functional_45_dense_89_matmul_readvariableop_resource:
6functional_45_dense_89_biasadd_readvariableop_resource:
6functional_45_conv2d_62_conv2d_readvariableop_resource;
7functional_45_conv2d_62_biasadd_readvariableop_resource9
5functional_45_dense_90_matmul_readvariableop_resource:
6functional_45_dense_90_biasadd_readvariableop_resource
identity??.functional_45/conv2d_60/BiasAdd/ReadVariableOp?-functional_45/conv2d_60/Conv2D/ReadVariableOp?.functional_45/conv2d_61/BiasAdd/ReadVariableOp?-functional_45/conv2d_61/Conv2D/ReadVariableOp?.functional_45/conv2d_62/BiasAdd/ReadVariableOp?-functional_45/conv2d_62/Conv2D/ReadVariableOp?-functional_45/dense_86/BiasAdd/ReadVariableOp?,functional_45/dense_86/MatMul/ReadVariableOp?-functional_45/dense_87/BiasAdd/ReadVariableOp?,functional_45/dense_87/MatMul/ReadVariableOp?-functional_45/dense_88/BiasAdd/ReadVariableOp?,functional_45/dense_88/MatMul/ReadVariableOp?-functional_45/dense_89/BiasAdd/ReadVariableOp?,functional_45/dense_89/MatMul/ReadVariableOp?-functional_45/dense_90/BiasAdd/ReadVariableOp?,functional_45/dense_90/MatMul/ReadVariableOp?
-functional_45/conv2d_60/Conv2D/ReadVariableOpReadVariableOp6functional_45_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_45/conv2d_60/Conv2D/ReadVariableOp?
functional_45/conv2d_60/Conv2DConv2Dinput_245functional_45/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2 
functional_45/conv2d_60/Conv2D?
.functional_45/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp7functional_45_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_45/conv2d_60/BiasAdd/ReadVariableOp?
functional_45/conv2d_60/BiasAddBiasAdd'functional_45/conv2d_60/Conv2D:output:06functional_45/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2!
functional_45/conv2d_60/BiasAdd?
functional_45/flatten_44/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2 
functional_45/flatten_44/Const?
 functional_45/flatten_44/ReshapeReshape(functional_45/conv2d_60/BiasAdd:output:0'functional_45/flatten_44/Const:output:0*
T0*)
_output_shapes
:???????????2"
 functional_45/flatten_44/Reshape?
,functional_45/dense_86/MatMul/ReadVariableOpReadVariableOp5functional_45_dense_86_matmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype02.
,functional_45/dense_86/MatMul/ReadVariableOp?
functional_45/dense_86/MatMulMatMul)functional_45/flatten_44/Reshape:output:04functional_45/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_86/MatMul?
-functional_45/dense_86/BiasAdd/ReadVariableOpReadVariableOp6functional_45_dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-functional_45/dense_86/BiasAdd/ReadVariableOp?
functional_45/dense_86/BiasAddBiasAdd'functional_45/dense_86/MatMul:product:05functional_45/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
functional_45/dense_86/BiasAdd?
functional_45/dense_86/ReluRelu'functional_45/dense_86/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_86/Relu?
!functional_45/dropout_56/IdentityIdentity)functional_45/dense_86/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!functional_45/dropout_56/Identity?
,functional_45/dense_87/MatMul/ReadVariableOpReadVariableOp5functional_45_dense_87_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,functional_45/dense_87/MatMul/ReadVariableOp?
functional_45/dense_87/MatMulMatMul*functional_45/dropout_56/Identity:output:04functional_45/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_87/MatMul?
-functional_45/dense_87/BiasAdd/ReadVariableOpReadVariableOp6functional_45_dense_87_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-functional_45/dense_87/BiasAdd/ReadVariableOp?
functional_45/dense_87/BiasAddBiasAdd'functional_45/dense_87/MatMul:product:05functional_45/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
functional_45/dense_87/BiasAdd?
functional_45/dense_87/ReluRelu'functional_45/dense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_87/Relu?
!functional_45/dropout_57/IdentityIdentity)functional_45/dense_87/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!functional_45/dropout_57/Identity?
functional_45/reshape_22/ShapeShape*functional_45/dropout_57/Identity:output:0*
T0*
_output_shapes
:2 
functional_45/reshape_22/Shape?
,functional_45/reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,functional_45/reshape_22/strided_slice/stack?
.functional_45/reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_45/reshape_22/strided_slice/stack_1?
.functional_45/reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_45/reshape_22/strided_slice/stack_2?
&functional_45/reshape_22/strided_sliceStridedSlice'functional_45/reshape_22/Shape:output:05functional_45/reshape_22/strided_slice/stack:output:07functional_45/reshape_22/strided_slice/stack_1:output:07functional_45/reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&functional_45/reshape_22/strided_slice?
(functional_45/reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2*
(functional_45/reshape_22/Reshape/shape/1?
(functional_45/reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2*
(functional_45/reshape_22/Reshape/shape/2?
(functional_45/reshape_22/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(functional_45/reshape_22/Reshape/shape/3?
&functional_45/reshape_22/Reshape/shapePack/functional_45/reshape_22/strided_slice:output:01functional_45/reshape_22/Reshape/shape/1:output:01functional_45/reshape_22/Reshape/shape/2:output:01functional_45/reshape_22/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&functional_45/reshape_22/Reshape/shape?
 functional_45/reshape_22/ReshapeReshape*functional_45/dropout_57/Identity:output:0/functional_45/reshape_22/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2"
 functional_45/reshape_22/Reshape?
-functional_45/conv2d_61/Conv2D/ReadVariableOpReadVariableOp6functional_45_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_45/conv2d_61/Conv2D/ReadVariableOp?
functional_45/conv2d_61/Conv2DConv2D)functional_45/reshape_22/Reshape:output:05functional_45/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2 
functional_45/conv2d_61/Conv2D?
.functional_45/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp7functional_45_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_45/conv2d_61/BiasAdd/ReadVariableOp?
functional_45/conv2d_61/BiasAddBiasAdd'functional_45/conv2d_61/Conv2D:output:06functional_45/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2!
functional_45/conv2d_61/BiasAdd?
functional_45/flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2 
functional_45/flatten_45/Const?
 functional_45/flatten_45/ReshapeReshape(functional_45/conv2d_61/BiasAdd:output:0'functional_45/flatten_45/Const:output:0*
T0*(
_output_shapes
:??????????2"
 functional_45/flatten_45/Reshape?
,functional_45/dense_88/MatMul/ReadVariableOpReadVariableOp5functional_45_dense_88_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02.
,functional_45/dense_88/MatMul/ReadVariableOp?
functional_45/dense_88/MatMulMatMul)functional_45/flatten_45/Reshape:output:04functional_45/dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_88/MatMul?
-functional_45/dense_88/BiasAdd/ReadVariableOpReadVariableOp6functional_45_dense_88_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-functional_45/dense_88/BiasAdd/ReadVariableOp?
functional_45/dense_88/BiasAddBiasAdd'functional_45/dense_88/MatMul:product:05functional_45/dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
functional_45/dense_88/BiasAdd?
functional_45/dense_88/ReluRelu'functional_45/dense_88/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_88/Relu?
!functional_45/dropout_58/IdentityIdentity)functional_45/dense_88/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!functional_45/dropout_58/Identity?
,functional_45/dense_89/MatMul/ReadVariableOpReadVariableOp5functional_45_dense_89_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,functional_45/dense_89/MatMul/ReadVariableOp?
functional_45/dense_89/MatMulMatMul*functional_45/dropout_58/Identity:output:04functional_45/dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_89/MatMul?
-functional_45/dense_89/BiasAdd/ReadVariableOpReadVariableOp6functional_45_dense_89_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-functional_45/dense_89/BiasAdd/ReadVariableOp?
functional_45/dense_89/BiasAddBiasAdd'functional_45/dense_89/MatMul:product:05functional_45/dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
functional_45/dense_89/BiasAdd?
functional_45/dense_89/ReluRelu'functional_45/dense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
functional_45/dense_89/Relu?
!functional_45/dropout_59/IdentityIdentity)functional_45/dense_89/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!functional_45/dropout_59/Identity?
functional_45/reshape_23/ShapeShape*functional_45/dropout_59/Identity:output:0*
T0*
_output_shapes
:2 
functional_45/reshape_23/Shape?
,functional_45/reshape_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,functional_45/reshape_23/strided_slice/stack?
.functional_45/reshape_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_45/reshape_23/strided_slice/stack_1?
.functional_45/reshape_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_45/reshape_23/strided_slice/stack_2?
&functional_45/reshape_23/strided_sliceStridedSlice'functional_45/reshape_23/Shape:output:05functional_45/reshape_23/strided_slice/stack:output:07functional_45/reshape_23/strided_slice/stack_1:output:07functional_45/reshape_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&functional_45/reshape_23/strided_slice?
(functional_45/reshape_23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2*
(functional_45/reshape_23/Reshape/shape/1?
(functional_45/reshape_23/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2*
(functional_45/reshape_23/Reshape/shape/2?
(functional_45/reshape_23/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(functional_45/reshape_23/Reshape/shape/3?
&functional_45/reshape_23/Reshape/shapePack/functional_45/reshape_23/strided_slice:output:01functional_45/reshape_23/Reshape/shape/1:output:01functional_45/reshape_23/Reshape/shape/2:output:01functional_45/reshape_23/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&functional_45/reshape_23/Reshape/shape?
 functional_45/reshape_23/ReshapeReshape*functional_45/dropout_59/Identity:output:0/functional_45/reshape_23/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2"
 functional_45/reshape_23/Reshape?
-functional_45/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6functional_45_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_45/conv2d_62/Conv2D/ReadVariableOp?
functional_45/conv2d_62/Conv2DConv2D)functional_45/reshape_23/Reshape:output:05functional_45/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2 
functional_45/conv2d_62/Conv2D?
.functional_45/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7functional_45_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_45/conv2d_62/BiasAdd/ReadVariableOp?
functional_45/conv2d_62/BiasAddBiasAdd'functional_45/conv2d_62/Conv2D:output:06functional_45/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2!
functional_45/conv2d_62/BiasAdd?
*functional_45/average_pooling2d_10/AvgPoolAvgPool(functional_45/conv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2,
*functional_45/average_pooling2d_10/AvgPool?
functional_45/flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????}   2 
functional_45/flatten_46/Const?
 functional_45/flatten_46/ReshapeReshape3functional_45/average_pooling2d_10/AvgPool:output:0'functional_45/flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????}2"
 functional_45/flatten_46/Reshape?
,functional_45/dense_90/MatMul/ReadVariableOpReadVariableOp5functional_45_dense_90_matmul_readvariableop_resource*
_output_shapes

:}*
dtype02.
,functional_45/dense_90/MatMul/ReadVariableOp?
functional_45/dense_90/MatMulMatMul)functional_45/flatten_46/Reshape:output:04functional_45/dense_90/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_45/dense_90/MatMul?
-functional_45/dense_90/BiasAdd/ReadVariableOpReadVariableOp6functional_45_dense_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_45/dense_90/BiasAdd/ReadVariableOp?
functional_45/dense_90/BiasAddBiasAdd'functional_45/dense_90/MatMul:product:05functional_45/dense_90/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
functional_45/dense_90/BiasAdd?
functional_45/dense_90/SoftmaxSoftmax'functional_45/dense_90/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
functional_45/dense_90/Softmax?
IdentityIdentity(functional_45/dense_90/Softmax:softmax:0/^functional_45/conv2d_60/BiasAdd/ReadVariableOp.^functional_45/conv2d_60/Conv2D/ReadVariableOp/^functional_45/conv2d_61/BiasAdd/ReadVariableOp.^functional_45/conv2d_61/Conv2D/ReadVariableOp/^functional_45/conv2d_62/BiasAdd/ReadVariableOp.^functional_45/conv2d_62/Conv2D/ReadVariableOp.^functional_45/dense_86/BiasAdd/ReadVariableOp-^functional_45/dense_86/MatMul/ReadVariableOp.^functional_45/dense_87/BiasAdd/ReadVariableOp-^functional_45/dense_87/MatMul/ReadVariableOp.^functional_45/dense_88/BiasAdd/ReadVariableOp-^functional_45/dense_88/MatMul/ReadVariableOp.^functional_45/dense_89/BiasAdd/ReadVariableOp-^functional_45/dense_89/MatMul/ReadVariableOp.^functional_45/dense_90/BiasAdd/ReadVariableOp-^functional_45/dense_90/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2`
.functional_45/conv2d_60/BiasAdd/ReadVariableOp.functional_45/conv2d_60/BiasAdd/ReadVariableOp2^
-functional_45/conv2d_60/Conv2D/ReadVariableOp-functional_45/conv2d_60/Conv2D/ReadVariableOp2`
.functional_45/conv2d_61/BiasAdd/ReadVariableOp.functional_45/conv2d_61/BiasAdd/ReadVariableOp2^
-functional_45/conv2d_61/Conv2D/ReadVariableOp-functional_45/conv2d_61/Conv2D/ReadVariableOp2`
.functional_45/conv2d_62/BiasAdd/ReadVariableOp.functional_45/conv2d_62/BiasAdd/ReadVariableOp2^
-functional_45/conv2d_62/Conv2D/ReadVariableOp-functional_45/conv2d_62/Conv2D/ReadVariableOp2^
-functional_45/dense_86/BiasAdd/ReadVariableOp-functional_45/dense_86/BiasAdd/ReadVariableOp2\
,functional_45/dense_86/MatMul/ReadVariableOp,functional_45/dense_86/MatMul/ReadVariableOp2^
-functional_45/dense_87/BiasAdd/ReadVariableOp-functional_45/dense_87/BiasAdd/ReadVariableOp2\
,functional_45/dense_87/MatMul/ReadVariableOp,functional_45/dense_87/MatMul/ReadVariableOp2^
-functional_45/dense_88/BiasAdd/ReadVariableOp-functional_45/dense_88/BiasAdd/ReadVariableOp2\
,functional_45/dense_88/MatMul/ReadVariableOp,functional_45/dense_88/MatMul/ReadVariableOp2^
-functional_45/dense_89/BiasAdd/ReadVariableOp-functional_45/dense_89/BiasAdd/ReadVariableOp2\
,functional_45/dense_89/MatMul/ReadVariableOp,functional_45/dense_89/MatMul/ReadVariableOp2^
-functional_45/dense_90/BiasAdd/ReadVariableOp-functional_45/dense_90/BiasAdd/ReadVariableOp2\
,functional_45/dense_90/MatMul/ReadVariableOp,functional_45/dense_90/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?o
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1405

inputs
conv2d_60_1330
conv2d_60_1332
dense_86_1336
dense_86_1338
dense_87_1342
dense_87_1344
conv2d_61_1349
conv2d_61_1351
dense_88_1355
dense_88_1357
dense_89_1361
dense_89_1363
conv2d_62_1368
conv2d_62_1370
dense_90_1375
dense_90_1377
identity??!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall? dense_86/StatefulPartitionedCall?1dense_86/kernel/Regularizer/Square/ReadVariableOp? dense_87/StatefulPartitionedCall?1dense_87/kernel/Regularizer/Square/ReadVariableOp? dense_88/StatefulPartitionedCall?1dense_88/kernel/Regularizer/Square/ReadVariableOp? dense_89/StatefulPartitionedCall?1dense_89/kernel/Regularizer/Square/ReadVariableOp? dense_90/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_1330conv2d_60_1332*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_60_layer_call_and_return_conditional_losses_6722#
!conv2d_60/StatefulPartitionedCall?
flatten_44/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_44_layer_call_and_return_conditional_losses_6942
flatten_44/PartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCall#flatten_44/PartitionedCall:output:0dense_86_1336dense_86_1338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_86_layer_call_and_return_conditional_losses_7192"
 dense_86/StatefulPartitionedCall?
dropout_56/PartitionedCallPartitionedCall)dense_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7522
dropout_56/PartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#dropout_56/PartitionedCall:output:0dense_87_1342dense_87_1344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_87_layer_call_and_return_conditional_losses_7822"
 dense_87/StatefulPartitionedCall?
dropout_57/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8152
dropout_57/PartitionedCall?
reshape_22/PartitionedCallPartitionedCall#dropout_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_22_layer_call_and_return_conditional_losses_8422
reshape_22/PartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall#reshape_22/PartitionedCall:output:0conv2d_61_1349conv2d_61_1351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_61_layer_call_and_return_conditional_losses_8602#
!conv2d_61/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_45_layer_call_and_return_conditional_losses_8822
flatten_45/PartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCall#flatten_45/PartitionedCall:output:0dense_88_1355dense_88_1357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_88_layer_call_and_return_conditional_losses_9072"
 dense_88/StatefulPartitionedCall?
dropout_58/PartitionedCallPartitionedCall)dense_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9402
dropout_58/PartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0dense_89_1361dense_89_1363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_89_layer_call_and_return_conditional_losses_9702"
 dense_89/StatefulPartitionedCall?
dropout_59/PartitionedCallPartitionedCall)dense_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_59_layer_call_and_return_conditional_losses_10032
dropout_59/PartitionedCall?
reshape_23/PartitionedCallPartitionedCall#dropout_59/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_23_layer_call_and_return_conditional_losses_10302
reshape_23/PartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall#reshape_23/PartitionedCall:output:0conv2d_62_1368conv2d_62_1370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_62_layer_call_and_return_conditional_losses_10482#
!conv2d_62/StatefulPartitionedCall?
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_6522&
$average_pooling2d_10/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????}* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_46_layer_call_and_return_conditional_losses_10712
flatten_46/PartitionedCall?
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_46/PartitionedCall:output:0dense_90_1375dense_90_1377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_90_layer_call_and_return_conditional_losses_10902"
 dense_90/StatefulPartitionedCall?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_86_1336* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_1342*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_88_1355*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_89_1361*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall2^dense_86/kernel/Regularizer/Square/ReadVariableOp!^dense_87/StatefulPartitionedCall2^dense_87/kernel/Regularizer/Square/ReadVariableOp!^dense_88/StatefulPartitionedCall2^dense_88/kernel/Regularizer/Square/ReadVariableOp!^dense_89/StatefulPartitionedCall2^dense_89/kernel/Regularizer/Square/ReadVariableOp!^dense_90/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

b
C__inference_dropout_59_layer_call_and_return_conditional_losses_998

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
`
D__inference_flatten_46_layer_call_and_return_conditional_losses_1071

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????}   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????}2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_56_layer_call_fn_1921

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
N
2__inference_average_pooling2d_10_layer_call_fn_658

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_6522
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?o
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1209
input_24
conv2d_60_1134
conv2d_60_1136
dense_86_1140
dense_86_1142
dense_87_1146
dense_87_1148
conv2d_61_1153
conv2d_61_1155
dense_88_1159
dense_88_1161
dense_89_1165
dense_89_1167
conv2d_62_1172
conv2d_62_1174
dense_90_1179
dense_90_1181
identity??!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall? dense_86/StatefulPartitionedCall?1dense_86/kernel/Regularizer/Square/ReadVariableOp? dense_87/StatefulPartitionedCall?1dense_87/kernel/Regularizer/Square/ReadVariableOp? dense_88/StatefulPartitionedCall?1dense_88/kernel/Regularizer/Square/ReadVariableOp? dense_89/StatefulPartitionedCall?1dense_89/kernel/Regularizer/Square/ReadVariableOp? dense_90/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinput_24conv2d_60_1134conv2d_60_1136*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_60_layer_call_and_return_conditional_losses_6722#
!conv2d_60/StatefulPartitionedCall?
flatten_44/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_44_layer_call_and_return_conditional_losses_6942
flatten_44/PartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCall#flatten_44/PartitionedCall:output:0dense_86_1140dense_86_1142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_86_layer_call_and_return_conditional_losses_7192"
 dense_86/StatefulPartitionedCall?
dropout_56/PartitionedCallPartitionedCall)dense_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_56_layer_call_and_return_conditional_losses_7522
dropout_56/PartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#dropout_56/PartitionedCall:output:0dense_87_1146dense_87_1148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_87_layer_call_and_return_conditional_losses_7822"
 dense_87/StatefulPartitionedCall?
dropout_57/PartitionedCallPartitionedCall)dense_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8152
dropout_57/PartitionedCall?
reshape_22/PartitionedCallPartitionedCall#dropout_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_22_layer_call_and_return_conditional_losses_8422
reshape_22/PartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall#reshape_22/PartitionedCall:output:0conv2d_61_1153conv2d_61_1155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_61_layer_call_and_return_conditional_losses_8602#
!conv2d_61/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_45_layer_call_and_return_conditional_losses_8822
flatten_45/PartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCall#flatten_45/PartitionedCall:output:0dense_88_1159dense_88_1161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_88_layer_call_and_return_conditional_losses_9072"
 dense_88/StatefulPartitionedCall?
dropout_58/PartitionedCallPartitionedCall)dense_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_58_layer_call_and_return_conditional_losses_9402
dropout_58/PartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0dense_89_1165dense_89_1167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_89_layer_call_and_return_conditional_losses_9702"
 dense_89/StatefulPartitionedCall?
dropout_59/PartitionedCallPartitionedCall)dense_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_59_layer_call_and_return_conditional_losses_10032
dropout_59/PartitionedCall?
reshape_23/PartitionedCallPartitionedCall#dropout_59/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_23_layer_call_and_return_conditional_losses_10302
reshape_23/PartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall#reshape_23/PartitionedCall:output:0conv2d_62_1172conv2d_62_1174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_62_layer_call_and_return_conditional_losses_10482#
!conv2d_62/StatefulPartitionedCall?
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_6522&
$average_pooling2d_10/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????}* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_46_layer_call_and_return_conditional_losses_10712
flatten_46/PartitionedCall?
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_46/PartitionedCall:output:0dense_90_1179dense_90_1181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_90_layer_call_and_return_conditional_losses_10902"
 dense_90/StatefulPartitionedCall?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_86_1140* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_87_1146*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_88_1159*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_89_1165*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall2^dense_86/kernel/Regularizer/Square/ReadVariableOp!^dense_87/StatefulPartitionedCall2^dense_87/kernel/Regularizer/Square/ReadVariableOp!^dense_88/StatefulPartitionedCall2^dense_88/kernel/Regularizer/Square/ReadVariableOp!^dense_89/StatefulPartitionedCall2^dense_89/kernel/Regularizer/Square/ReadVariableOp!^dense_90/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?
`
D__inference_reshape_23_layer_call_and_return_conditional_losses_2166

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????

2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
,__inference_functional_45_layer_call_fn_1440
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_45_layer_call_and_return_conditional_losses_14052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?
b
D__inference_dropout_59_layer_call_and_return_conditional_losses_1003

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_2232>
:dense_86_kernel_regularizer_square_readvariableop_resource
identity??1dense_86/kernel/Regularizer/Square/ReadVariableOp?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_86_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
IdentityIdentity#dense_86/kernel/Regularizer/mul:z:02^dense_86/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_3_2265>
:dense_89_kernel_regularizer_square_readvariableop_resource
identity??1dense_89/kernel/Regularizer/Square/ReadVariableOp?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_89_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentity#dense_89/kernel/Regularizer/mul:z:02^dense_89/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp
?
|
'__inference_dense_90_layer_call_fn_2221

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_90_layer_call_and_return_conditional_losses_10902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????}::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????}
 
_user_specified_nameinputs
?
b
)__inference_dropout_57_layer_call_fn_1980

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_57_layer_call_and_return_conditional_losses_8102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_flatten_46_layer_call_fn_2201

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????}* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_46_layer_call_and_return_conditional_losses_10712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_59_layer_call_fn_2152

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_59_layer_call_and_return_conditional_losses_10032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
C__inference_dropout_58_layer_call_and_return_conditional_losses_940

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_flatten_44_layer_call_fn_1867

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_44_layer_call_and_return_conditional_losses_6942
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_56_layer_call_and_return_conditional_losses_1916

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_61_layer_call_and_return_conditional_losses_2014

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
c
D__inference_dropout_59_layer_call_and_return_conditional_losses_2137

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
A__inference_dense_86_layer_call_and_return_conditional_losses_719

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_86/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_86/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
i
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_652

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_58_layer_call_and_return_conditional_losses_2078

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
_
C__inference_flatten_44_layer_call_and_return_conditional_losses_694

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_dense_89_layer_call_and_return_conditional_losses_970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_89/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_89/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
c
D__inference_dropout_56_layer_call_and_return_conditional_losses_1911

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
"__inference_signature_wrapper_1511
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_6462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_24
?
E
)__inference_reshape_22_layer_call_fn_2004

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_22_layer_call_and_return_conditional_losses_8422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
C__inference_dropout_56_layer_call_and_return_conditional_losses_752

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
G__inference_functional_45_layer_call_and_return_conditional_losses_1763

inputs,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource,
(conv2d_61_conv2d_readvariableop_resource-
)conv2d_61_biasadd_readvariableop_resource+
'dense_88_matmul_readvariableop_resource,
(dense_88_biasadd_readvariableop_resource+
'dense_89_matmul_readvariableop_resource,
(dense_89_biasadd_readvariableop_resource,
(conv2d_62_conv2d_readvariableop_resource-
)conv2d_62_biasadd_readvariableop_resource+
'dense_90_matmul_readvariableop_resource,
(dense_90_biasadd_readvariableop_resource
identity?? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp?dense_86/BiasAdd/ReadVariableOp?dense_86/MatMul/ReadVariableOp?1dense_86/kernel/Regularizer/Square/ReadVariableOp?dense_87/BiasAdd/ReadVariableOp?dense_87/MatMul/ReadVariableOp?1dense_87/kernel/Regularizer/Square/ReadVariableOp?dense_88/BiasAdd/ReadVariableOp?dense_88/MatMul/ReadVariableOp?1dense_88/kernel/Regularizer/Square/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?1dense_89/kernel/Regularizer/Square/ReadVariableOp?dense_90/BiasAdd/ReadVariableOp?dense_90/MatMul/ReadVariableOp?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_60/BiasAddu
flatten_44/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_44/Const?
flatten_44/ReshapeReshapeconv2d_60/BiasAdd:output:0flatten_44/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_44/Reshape?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulflatten_44/Reshape:output:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_86/MatMul?
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_86/BiasAdd/ReadVariableOp?
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_86/Relu?
dropout_56/IdentityIdentitydense_86/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
dropout_56/Identity?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMuldropout_56/Identity:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_87/MatMul?
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_87/BiasAdd/ReadVariableOp?
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_87/BiasAdds
dense_87/ReluReludense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_87/Relu?
dropout_57/IdentityIdentitydense_87/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
dropout_57/Identityp
reshape_22/ShapeShapedropout_57/Identity:output:0*
T0*
_output_shapes
:2
reshape_22/Shape?
reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_22/strided_slice/stack?
 reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_22/strided_slice/stack_1?
 reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_22/strided_slice/stack_2?
reshape_22/strided_sliceStridedSlicereshape_22/Shape:output:0'reshape_22/strided_slice/stack:output:0)reshape_22/strided_slice/stack_1:output:0)reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_22/strided_slicez
reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_22/Reshape/shape/1z
reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_22/Reshape/shape/2z
reshape_22/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_22/Reshape/shape/3?
reshape_22/Reshape/shapePack!reshape_22/strided_slice:output:0#reshape_22/Reshape/shape/1:output:0#reshape_22/Reshape/shape/2:output:0#reshape_22/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_22/Reshape/shape?
reshape_22/ReshapeReshapedropout_57/Identity:output:0!reshape_22/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2
reshape_22/Reshape?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_61/Conv2D/ReadVariableOp?
conv2d_61/Conv2DConv2Dreshape_22/Reshape:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
conv2d_61/Conv2D?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
conv2d_61/BiasAddu
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_45/Const?
flatten_45/ReshapeReshapeconv2d_61/BiasAdd:output:0flatten_45/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_45/Reshape?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulflatten_45/Reshape:output:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_88/MatMul?
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_88/BiasAdd/ReadVariableOp?
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_88/BiasAdds
dense_88/ReluReludense_88/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_88/Relu?
dropout_58/IdentityIdentitydense_88/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
dropout_58/Identity?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMuldropout_58/Identity:output:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_89/BiasAdds
dense_89/ReluReludense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_89/Relu?
dropout_59/IdentityIdentitydense_89/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
dropout_59/Identityp
reshape_23/ShapeShapedropout_59/Identity:output:0*
T0*
_output_shapes
:2
reshape_23/Shape?
reshape_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_23/strided_slice/stack?
 reshape_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_23/strided_slice/stack_1?
 reshape_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_23/strided_slice/stack_2?
reshape_23/strided_sliceStridedSlicereshape_23/Shape:output:0'reshape_23/strided_slice/stack:output:0)reshape_23/strided_slice/stack_1:output:0)reshape_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_23/strided_slicez
reshape_23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_23/Reshape/shape/1z
reshape_23/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_23/Reshape/shape/2z
reshape_23/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_23/Reshape/shape/3?
reshape_23/Reshape/shapePack!reshape_23/strided_slice:output:0#reshape_23/Reshape/shape/1:output:0#reshape_23/Reshape/shape/2:output:0#reshape_23/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_23/Reshape/shape?
reshape_23/ReshapeReshapedropout_59/Identity:output:0!reshape_23/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????

2
reshape_23/Reshape?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2Dreshape_23/Reshape:output:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingSAME*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
conv2d_62/BiasAdd?
average_pooling2d_10/AvgPoolAvgPoolconv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
average_pooling2d_10/AvgPoolu
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????}   2
flatten_46/Const?
flatten_46/ReshapeReshape%average_pooling2d_10/AvgPool:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????}2
flatten_46/Reshape?
dense_90/MatMul/ReadVariableOpReadVariableOp'dense_90_matmul_readvariableop_resource*
_output_shapes

:}*
dtype02 
dense_90/MatMul/ReadVariableOp?
dense_90/MatMulMatMulflatten_46/Reshape:output:0&dense_90/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_90/MatMul?
dense_90/BiasAdd/ReadVariableOpReadVariableOp(dense_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_90/BiasAdd/ReadVariableOp?
dense_90/BiasAddBiasAdddense_90/MatMul:product:0'dense_90/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_90/BiasAdd|
dense_90/SoftmaxSoftmaxdense_90/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_90/Softmax?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
1dense_87/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_87/kernel/Regularizer/Square/ReadVariableOp?
"dense_87/kernel/Regularizer/SquareSquare9dense_87/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_87/kernel/Regularizer/Square?
!dense_87/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_87/kernel/Regularizer/Const?
dense_87/kernel/Regularizer/SumSum&dense_87/kernel/Regularizer/Square:y:0*dense_87/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/Sum?
!dense_87/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_87/kernel/Regularizer/mul/x?
dense_87/kernel/Regularizer/mulMul*dense_87/kernel/Regularizer/mul/x:output:0(dense_87/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_87/kernel/Regularizer/mul?
1dense_88/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype023
1dense_88/kernel/Regularizer/Square/ReadVariableOp?
"dense_88/kernel/Regularizer/SquareSquare9dense_88/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d2$
"dense_88/kernel/Regularizer/Square?
!dense_88/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_88/kernel/Regularizer/Const?
dense_88/kernel/Regularizer/SumSum&dense_88/kernel/Regularizer/Square:y:0*dense_88/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/Sum?
!dense_88/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_88/kernel/Regularizer/mul/x?
dense_88/kernel/Regularizer/mulMul*dense_88/kernel/Regularizer/mul/x:output:0(dense_88/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_88/kernel/Regularizer/mul?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentitydense_90/Softmax:softmax:0!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp2^dense_86/kernel/Regularizer/Square/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp2^dense_87/kernel/Regularizer/Square/ReadVariableOp ^dense_88/BiasAdd/ReadVariableOp^dense_88/MatMul/ReadVariableOp2^dense_88/kernel/Regularizer/Square/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp2^dense_89/kernel/Regularizer/Square/ReadVariableOp ^dense_90/BiasAdd/ReadVariableOp^dense_90/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2f
1dense_87/kernel/Regularizer/Square/ReadVariableOp1dense_87/kernel/Regularizer/Square/ReadVariableOp2B
dense_88/BiasAdd/ReadVariableOpdense_88/BiasAdd/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2f
1dense_88/kernel/Regularizer/Square/ReadVariableOp1dense_88/kernel/Regularizer/Square/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp2B
dense_90/BiasAdd/ReadVariableOpdense_90/BiasAdd/ReadVariableOp2@
dense_90/MatMul/ReadVariableOpdense_90/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_dense_86_layer_call_and_return_conditional_losses_1890

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_86/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_86/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??d*
dtype023
1dense_86/kernel/Regularizer/Square/ReadVariableOp?
"dense_86/kernel/Regularizer/SquareSquare9dense_86/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??d2$
"dense_86/kernel/Regularizer/Square?
!dense_86/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_86/kernel/Regularizer/Const?
dense_86/kernel/Regularizer/SumSum&dense_86/kernel/Regularizer/Square:y:0*dense_86/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/Sum?
!dense_86/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_86/kernel/Regularizer/mul/x?
dense_86/kernel/Regularizer/mulMul*dense_86/kernel/Regularizer/mul/x:output:0(dense_86/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_86/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_86/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_86/kernel/Regularizer/Square/ReadVariableOp1dense_86/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_dense_89_layer_call_and_return_conditional_losses_2116

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_89/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
1dense_89/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1dense_89/kernel/Regularizer/Square/ReadVariableOp?
"dense_89/kernel/Regularizer/SquareSquare9dense_89/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2$
"dense_89/kernel/Regularizer/Square?
!dense_89/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_89/kernel/Regularizer/Const?
dense_89/kernel/Regularizer/SumSum&dense_89/kernel/Regularizer/Square:y:0*dense_89/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/Sum?
!dense_89/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_89/kernel/Regularizer/mul/x?
dense_89/kernel/Regularizer/mulMul*dense_89/kernel/Regularizer/mul/x:output:0(dense_89/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_89/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_89/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_89/kernel/Regularizer/Square/ReadVariableOp1dense_89/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
b
D__inference_dropout_57_layer_call_and_return_conditional_losses_1975

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
input_24;
serving_default_input_24:0???????????<
dense_900
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ӆ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
layer-17
layer_with_weights-7
layer-18
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "functional_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_45", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 129, 180, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}, "name": "input_24", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_60", "inbound_nodes": [[["input_24", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_44", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_44", "inbound_nodes": [[["conv2d_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_86", "inbound_nodes": [[["flatten_44", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_56", "inbound_nodes": [[["dense_86", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_87", "inbound_nodes": [[["dropout_56", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_57", "inbound_nodes": [[["dense_87", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}, "name": "reshape_22", "inbound_nodes": [[["dropout_57", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_61", "inbound_nodes": [[["reshape_22", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_45", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_45", "inbound_nodes": [[["conv2d_61", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_88", "inbound_nodes": [[["flatten_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_58", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_58", "inbound_nodes": [[["dense_88", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_89", "inbound_nodes": [[["dropout_58", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_59", "inbound_nodes": [[["dense_89", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}, "name": "reshape_23", "inbound_nodes": [[["dropout_59", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_62", "inbound_nodes": [[["reshape_23", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d_10", "inbound_nodes": [[["conv2d_62", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_46", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_46", "inbound_nodes": [[["average_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_90", "inbound_nodes": [[["flatten_46", 0, 0, {}]]]}], "input_layers": [["input_24", 0, 0]], "output_layers": [["dense_90", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 129, 180, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 129, 180, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_45", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 129, 180, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}, "name": "input_24", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_60", "inbound_nodes": [[["input_24", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_44", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_44", "inbound_nodes": [[["conv2d_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_86", "inbound_nodes": [[["flatten_44", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_56", "inbound_nodes": [[["dense_86", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_87", "inbound_nodes": [[["dropout_56", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_57", "inbound_nodes": [[["dense_87", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}, "name": "reshape_22", "inbound_nodes": [[["dropout_57", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_61", "inbound_nodes": [[["reshape_22", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_45", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_45", "inbound_nodes": [[["conv2d_61", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_88", "inbound_nodes": [[["flatten_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_58", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_58", "inbound_nodes": [[["dense_88", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_89", "inbound_nodes": [[["dropout_58", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_59", "inbound_nodes": [[["dense_89", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}, "name": "reshape_23", "inbound_nodes": [[["dropout_59", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_62", "inbound_nodes": [[["reshape_23", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d_10", "inbound_nodes": [[["conv2d_62", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_46", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_46", "inbound_nodes": [[["average_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_90", "inbound_nodes": [[["flatten_46", 0, 0, {}]]]}], "input_layers": [["input_24", 0, 0]], "output_layers": [["dense_90", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_24", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 129, 180, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 129, 180, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 129, 180, 1]}}
?
 regularization_losses
!trainable_variables
"	variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_44", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 116100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 116100]}}
?
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_56", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_56", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_87", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_57", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}}
?	

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 1]}}
?
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_45", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_88", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
?
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_58", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_58", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_89", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 1]}}}
?	

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 1]}}
?
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "AveragePooling2D", "name": "average_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_46", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_90", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 125}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 125]}}
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m?$m?%m?.m?/m?<m?=m?Fm?Gm?Pm?Qm?^m?_m?lm?mm?v?v?$v?%v?.v?/v?<v?=v?Fv?Gv?Pv?Qv?^v?_v?lv?mv?"
	optimizer
@
?0
?1
?2
?3"
trackable_list_wrapper
?
0
1
$2
%3
.4
/5
<6
=7
F8
G9
P10
Q11
^12
_13
l14
m15"
trackable_list_wrapper
?
0
1
$2
%3
.4
/5
<6
=7
F8
G9
P10
Q11
^12
_13
l14
m15"
trackable_list_wrapper
?
regularization_losses
wlayer_metrics
xlayer_regularization_losses

ylayers
zmetrics
{non_trainable_variables
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_60/kernel
:2conv2d_60/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
|layer_metrics
}layer_regularization_losses

~layers
metrics
?non_trainable_variables
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
!trainable_variables
"	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??d2dense_86/kernel
:d2dense_86/bias
(
?0"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
&regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
'trainable_variables
(	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
+trainable_variables
,	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_87/kernel
:d2dense_87/bias
(
?0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
1trainable_variables
2	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
5trainable_variables
6	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
9trainable_variables
:	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_61/kernel
:2conv2d_61/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
>regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
?trainable_variables
@	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Ctrainable_variables
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?d2dense_88/kernel
:d2dense_88/bias
(
?0"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
Hregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Itrainable_variables
J	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Mtrainable_variables
N	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_89/kernel
:d2dense_89/bias
(
?0"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
Rregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Strainable_variables
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
Wtrainable_variables
X	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
[trainable_variables
\	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_62/kernel
:2conv2d_62/bias
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
`regularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
atrainable_variables
b	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
etrainable_variables
f	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
itrainable_variables
j	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:}2dense_90/kernel
:2dense_90/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?non_trainable_variables
otrainable_variables
p	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
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
17
18"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(2conv2d_60/kernel/m
:2conv2d_60/bias/m
#:!
??d2dense_86/kernel/m
:d2dense_86/bias/m
!:dd2dense_87/kernel/m
:d2dense_87/bias/m
*:(2conv2d_61/kernel/m
:2conv2d_61/bias/m
": 	?d2dense_88/kernel/m
:d2dense_88/bias/m
!:dd2dense_89/kernel/m
:d2dense_89/bias/m
*:(2conv2d_62/kernel/m
:2conv2d_62/bias/m
!:}2dense_90/kernel/m
:2dense_90/bias/m
*:(2conv2d_60/kernel/v
:2conv2d_60/bias/v
#:!
??d2dense_86/kernel/v
:d2dense_86/bias/v
!:dd2dense_87/kernel/v
:d2dense_87/bias/v
*:(2conv2d_61/kernel/v
:2conv2d_61/bias/v
": 	?d2dense_88/kernel/v
:d2dense_88/bias/v
!:dd2dense_89/kernel/v
:d2dense_89/bias/v
*:(2conv2d_62/kernel/v
:2conv2d_62/bias/v
!:}2dense_90/kernel/v
:2dense_90/bias/v
?2?
G__inference_functional_45_layer_call_and_return_conditional_losses_1651
G__inference_functional_45_layer_call_and_return_conditional_losses_1763
G__inference_functional_45_layer_call_and_return_conditional_losses_1131
G__inference_functional_45_layer_call_and_return_conditional_losses_1209?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_646?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
input_24???????????
?2?
,__inference_functional_45_layer_call_fn_1440
,__inference_functional_45_layer_call_fn_1800
,__inference_functional_45_layer_call_fn_1325
,__inference_functional_45_layer_call_fn_1837?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_60_layer_call_and_return_conditional_losses_1847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_60_layer_call_fn_1856?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_44_layer_call_and_return_conditional_losses_1862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_44_layer_call_fn_1867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_86_layer_call_and_return_conditional_losses_1890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_86_layer_call_fn_1899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_56_layer_call_and_return_conditional_losses_1916
D__inference_dropout_56_layer_call_and_return_conditional_losses_1911?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_56_layer_call_fn_1921
)__inference_dropout_56_layer_call_fn_1926?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_87_layer_call_and_return_conditional_losses_1949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_87_layer_call_fn_1958?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_57_layer_call_and_return_conditional_losses_1975
D__inference_dropout_57_layer_call_and_return_conditional_losses_1970?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_57_layer_call_fn_1985
)__inference_dropout_57_layer_call_fn_1980?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_reshape_22_layer_call_and_return_conditional_losses_1999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_22_layer_call_fn_2004?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_61_layer_call_and_return_conditional_losses_2014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_61_layer_call_fn_2023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_45_layer_call_and_return_conditional_losses_2029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_45_layer_call_fn_2034?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_88_layer_call_and_return_conditional_losses_2057?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_88_layer_call_fn_2066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_58_layer_call_and_return_conditional_losses_2083
D__inference_dropout_58_layer_call_and_return_conditional_losses_2078?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_58_layer_call_fn_2088
)__inference_dropout_58_layer_call_fn_2093?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_89_layer_call_and_return_conditional_losses_2116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_89_layer_call_fn_2125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_59_layer_call_and_return_conditional_losses_2142
D__inference_dropout_59_layer_call_and_return_conditional_losses_2137?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_59_layer_call_fn_2147
)__inference_dropout_59_layer_call_fn_2152?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_reshape_23_layer_call_and_return_conditional_losses_2166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_23_layer_call_fn_2171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_62_layer_call_and_return_conditional_losses_2181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_62_layer_call_fn_2190?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_average_pooling2d_10_layer_call_fn_658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_flatten_46_layer_call_and_return_conditional_losses_2196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_46_layer_call_fn_2201?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_90_layer_call_and_return_conditional_losses_2212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_90_layer_call_fn_2221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_2232?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_2243?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_2254?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_2265?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_1511input_24"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_646?$%./<=FGPQ^_lm;?8
1?.
,?)
input_24???????????
? "3?0
.
dense_90"?
dense_90??????????
M__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_652?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_average_pooling2d_10_layer_call_fn_658?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_conv2d_60_layer_call_and_return_conditional_losses_1847p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_60_layer_call_fn_1856c9?6
/?,
*?'
inputs???????????
? ""?????????????
C__inference_conv2d_61_layer_call_and_return_conditional_losses_2014l<=7?4
-?*
(?%
inputs?????????


? "-?*
#? 
0?????????


? ?
(__inference_conv2d_61_layer_call_fn_2023_<=7?4
-?*
(?%
inputs?????????


? " ??????????

?
C__inference_conv2d_62_layer_call_and_return_conditional_losses_2181l^_7?4
-?*
(?%
inputs?????????


? "-?*
#? 
0?????????


? ?
(__inference_conv2d_62_layer_call_fn_2190_^_7?4
-?*
(?%
inputs?????????


? " ??????????

?
B__inference_dense_86_layer_call_and_return_conditional_losses_1890^$%1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????d
? |
'__inference_dense_86_layer_call_fn_1899Q$%1?.
'?$
"?
inputs???????????
? "??????????d?
B__inference_dense_87_layer_call_and_return_conditional_losses_1949\.//?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? z
'__inference_dense_87_layer_call_fn_1958O.//?,
%?"
 ?
inputs?????????d
? "??????????d?
B__inference_dense_88_layer_call_and_return_conditional_losses_2057]FG0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? {
'__inference_dense_88_layer_call_fn_2066PFG0?-
&?#
!?
inputs??????????
? "??????????d?
B__inference_dense_89_layer_call_and_return_conditional_losses_2116\PQ/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? z
'__inference_dense_89_layer_call_fn_2125OPQ/?,
%?"
 ?
inputs?????????d
? "??????????d?
B__inference_dense_90_layer_call_and_return_conditional_losses_2212\lm/?,
%?"
 ?
inputs?????????}
? "%?"
?
0?????????
? z
'__inference_dense_90_layer_call_fn_2221Olm/?,
%?"
 ?
inputs?????????}
? "???????????
D__inference_dropout_56_layer_call_and_return_conditional_losses_1911\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
D__inference_dropout_56_layer_call_and_return_conditional_losses_1916\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? |
)__inference_dropout_56_layer_call_fn_1921O3?0
)?&
 ?
inputs?????????d
p
? "??????????d|
)__inference_dropout_56_layer_call_fn_1926O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
D__inference_dropout_57_layer_call_and_return_conditional_losses_1970\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
D__inference_dropout_57_layer_call_and_return_conditional_losses_1975\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? |
)__inference_dropout_57_layer_call_fn_1980O3?0
)?&
 ?
inputs?????????d
p
? "??????????d|
)__inference_dropout_57_layer_call_fn_1985O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
D__inference_dropout_58_layer_call_and_return_conditional_losses_2078\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
D__inference_dropout_58_layer_call_and_return_conditional_losses_2083\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? |
)__inference_dropout_58_layer_call_fn_2088O3?0
)?&
 ?
inputs?????????d
p
? "??????????d|
)__inference_dropout_58_layer_call_fn_2093O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
D__inference_dropout_59_layer_call_and_return_conditional_losses_2137\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
D__inference_dropout_59_layer_call_and_return_conditional_losses_2142\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? |
)__inference_dropout_59_layer_call_fn_2147O3?0
)?&
 ?
inputs?????????d
p
? "??????????d|
)__inference_dropout_59_layer_call_fn_2152O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
D__inference_flatten_44_layer_call_and_return_conditional_losses_1862d9?6
/?,
*?'
inputs???????????
? "'?$
?
0???????????
? ?
)__inference_flatten_44_layer_call_fn_1867W9?6
/?,
*?'
inputs???????????
? "?????????????
D__inference_flatten_45_layer_call_and_return_conditional_losses_2029a7?4
-?*
(?%
inputs?????????


? "&?#
?
0??????????
? ?
)__inference_flatten_45_layer_call_fn_2034T7?4
-?*
(?%
inputs?????????


? "????????????
D__inference_flatten_46_layer_call_and_return_conditional_losses_2196`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????}
? ?
)__inference_flatten_46_layer_call_fn_2201S7?4
-?*
(?%
inputs?????????
? "??????????}?
G__inference_functional_45_layer_call_and_return_conditional_losses_1131~$%./<=FGPQ^_lmC?@
9?6
,?)
input_24???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_functional_45_layer_call_and_return_conditional_losses_1209~$%./<=FGPQ^_lmC?@
9?6
,?)
input_24???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_functional_45_layer_call_and_return_conditional_losses_1651|$%./<=FGPQ^_lmA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_functional_45_layer_call_and_return_conditional_losses_1763|$%./<=FGPQ^_lmA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_functional_45_layer_call_fn_1325q$%./<=FGPQ^_lmC?@
9?6
,?)
input_24???????????
p

 
? "???????????
,__inference_functional_45_layer_call_fn_1440q$%./<=FGPQ^_lmC?@
9?6
,?)
input_24???????????
p 

 
? "???????????
,__inference_functional_45_layer_call_fn_1800o$%./<=FGPQ^_lmA?>
7?4
*?'
inputs???????????
p

 
? "???????????
,__inference_functional_45_layer_call_fn_1837o$%./<=FGPQ^_lmA?>
7?4
*?'
inputs???????????
p 

 
? "??????????9
__inference_loss_fn_0_2232$?

? 
? "? 9
__inference_loss_fn_1_2243.?

? 
? "? 9
__inference_loss_fn_2_2254F?

? 
? "? 9
__inference_loss_fn_3_2265P?

? 
? "? ?
D__inference_reshape_22_layer_call_and_return_conditional_losses_1999`/?,
%?"
 ?
inputs?????????d
? "-?*
#? 
0?????????


? ?
)__inference_reshape_22_layer_call_fn_2004S/?,
%?"
 ?
inputs?????????d
? " ??????????

?
D__inference_reshape_23_layer_call_and_return_conditional_losses_2166`/?,
%?"
 ?
inputs?????????d
? "-?*
#? 
0?????????


? ?
)__inference_reshape_23_layer_call_fn_2171S/?,
%?"
 ?
inputs?????????d
? " ??????????

?
"__inference_signature_wrapper_1511?$%./<=FGPQ^_lmG?D
? 
=?:
8
input_24,?)
input_24???????????"3?0
.
dense_90"?
dense_90?????????