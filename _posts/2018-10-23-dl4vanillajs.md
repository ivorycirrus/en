---
layout: post
title:  "A short trip to Deep-learning with Javascript"
date:   2018-10-23 13:00
categories: [javascript, machine learning]
permalink: /archivers/dl4vanillajs
lang: "en"
lang_default: "en"
lang_alternate: ["ko", "en"]
---

Nowadays, when we are studying Deep Learning or Machine Learning, we will try out of Python. And we can easily create machine-running or deep-running applications using the framework such as Tensorflow, Keras and pyTorch. In fact, in most cases, I agree these approaches. There are many features that need to be implemented so that you can make what you want quickly and easily, and it is easy to get help through web surfing or community even if you have problems.

But this post is started from some questions. 'Why should not it be Python?', 'Do I have to use the framework?' So I choose an easy-to-access scripting language - Javascript, because It runs on a web browser and on a stand-alone apllication. Of course, It will not use any framework, and we will build your own Deep-learning model with only basic language commands. And It will introduce the process of learning and evaluating the model for a simple example (although it can not guarantee performance).


## 1. Intro
Frankly, this post is not a great jobs such as a development guide or a best practices. That also does not mean that [CNN](http://cs231n.github.io/convolutional-networks/)/[RNN](https://towardsdatascience.com/recurrent-neural-networks-and-lstm-4b601dd822a5) and other famous artificial neural networks should be implemented. In this post, we will create a model with a simple level of [MLP(Multi-Layer Perceptron)](https://skymind.ai/wiki/multilayer-perceptron), and to train and evaluate through a sample project. Although it is a small project that started with a tremendous  motivation and a rudimentary goal, I would like you to look at it lightly with a courageous and challenging for creating a deep-learning application with just one scripting language.

The source code and example code are located in the Github link below.

* **[dl4vanilla.js](https://github.com/ivorycirrus/dl4vanillajs)** : Matrix operations and ANN creation utility written in Javascript
* **[dl4vanilla.js Examples](https://github.com/ivorycirrus/dl4vanillajs-node-example)** : A multilayer perceptron (MLP) example that can be run on Node.js.


## 2. Create the tools
A general deep-learning model involves estimating the results of a matrix operation and a training step of updating the parameters of the model with the error of the estimated result. The input values ​​in the model should be represented by numerical data in the form of a matrix, and then the result values ​​are estimated through various computation processes and variables of the matrix type in the model. And we aim to increase the accuracy of the estimated values by updating the model's parameters toward the direction in which the error is decreasing.

Therefore, in order to implement the process, the function of performing the matrix operation of the model's parameters should be prepared. And the Activation function is necessory In order to prevent the matrix operation from being a simple linear linear combination, it is necessary to add non-linearity between the matrix multiplications and additions which will prevent multi-layered networks abbreviated just one linear-combination of matrix. Below here, we will try to create the element functions to construct the model first.


### 2.1 Matrix operation
The general artificial neural network which consists of [Forward propagation](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/fc_layer.html) of [Fully Connected layers](https://www.bogotobogo.com/python/scikit-learn/Artificial-Neural-Network-ANN-2-Forward-Propagation.php) includes a process of multiplying an input matrix by the weight matrix and adding the bias matrix. Therefore, the operation of the matrix can be perform addition and multiplication of the matrix at least.

Below is a implementation of matrix addition. If the input matrix is ​​a one-dimensional matrix, it returns a one-dimensional matrix that is the sum of the matrix values. And input matrix is ​​a multi-dimensional matrix, it reduce matrix dimension and call recursively. As a result, we implemented a function that can successfully obtain the matrix addidion, even if the input matrix is deep in dimension, or even if it is not square matrices. Of course, if both of input matrixs have ​​different shape or data type is different, it occurs an exception.

```javascript
// [[ math/matrix.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/math/matrix.js

/* Matrix addition */
let _matrix_add = function(arr1, arr2){
	if(!Array.isArray(arr1) || !Array.isArray(arr2)) throw "MatrixException : parameters are not array.";
	else if(arr1.length != arr2.length) throw "MatrixException : Size of arrays are different.";
	else if(arr1.length == 0) return null;
	else {
		let result = [];
		for(let i = 0 ; i < arr1.length ; i++) {
			if(Array.isArray(arr1[i])) result[i] = _matrix_add(arr1[i], arr2[i]);
			else result[i] = arr1[i] + arr2[i];
		}
		return result;
	}
};
```

The multiplication of the matrix works only two-dimensional matrices. If you need more than three or higher dimensional matrix multiplication, you may apply [Tensor Contraction](https://math.stackexchange.com/a/63139). But I think this two-dimensional matrix multiplication is good enough to handle simple single-channel data.

```javascript
// [[ math/matrix.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/math/matrix.js

/* Matrix multiply */
let _matrix_mul = function(arr1, arr2){
	if(!Array.isArray(arr1) || !Array.isArray(arr2)) throw "MatrixException : parameters are not array.";

	const s1 = _matrix_shape(arr1, []);
	const s2 = _matrix_shape(arr2, []);
	if(s1.length != 2 || s2.length != 2) throw "MatrixException : input arrays are not 2d array.";
	else if(s1[1] != s2[0]) throw "MatrixException : array shapes are mismatch.";
	else if(s1[0] == 0 || s2[1] == 0) throw "MatrixException : cannot multiple zseo-size array.";

	const rows = s1[0], cols = s2[1], middle = s1[1];
	let result = [];
	for(let i = 0 ; i < rows ; i++) {
		let row = [];
		for(let j = 0 ; j < cols ; j++) {
			let cell = 0;
			for(let k = 0 ; k < middle ; k++) {
				cell += (arr1[i][k] * arr2[k][j])
			}
			row[j] = cell;
		}
		result[i] = row;
	}

	return result;
};
```

In addition, we have some other utilities such as the function can apply operations on each element value of a matrix. This may be useful for setting the initial value of the matrix or performing normalization on the matrix values. The parameters are one array and one function. It invocks the function by each values of the matrix, and composes new matrix as return.

```javascript
// [[ math/matrix.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/math/matrix.js

/* Evaluate function */
let _eval_mat = function(arr1, func) {
	if(!Array.isArray(arr1)) throw "MatrixException : first parameter is not array.";
	else if(typeof func != `function`) throw "MatrixException : second parameter is not function.";
	else {
		let mapper = x => Array.isArray(x)?x.map(mapper):func(x);
		return arr1.map(mapper);
	}
}
```

There are also a function to find a transposed matrix, a function to obtain and change the shape of a matrix, and a function to find the maximum / minimum value of elements in a matrix. These functions are not a key component of learning and estimation in the model, but I think they can be useful for processing input or output values.


### 2.2 Activation function
The computation of the artificial neural network consists of sum and multiplication of matrices. However, the sum and multiplication of simple matrices can only be expressed in one form of a simple linear transformation, even if the operations are performed multiple times. This means that connecting neural networks many times is difficult to achieve by normal ways. The [activation function](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) plays a role in preventing the data to simplified in the form of linear transformation by adding nonlinearity to the operations of the matrices.

Well-known activation functions are [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function). We are going to implement for simple classification codes, so Sigmoid and ReLU functions are prepared. First, we define an activation function that can handle scalar values, and we apply the activation function to each element of the matrix using `eval(arr,func)` among the functions prepared for matrix operation.

```javascript
// [[ nn/activation_function.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/nn/activation_function.js

/* Sigmoid(x) = 1/(1+exp(-x))*/
const _func_sigmoid = function(x){return 1.0/(1.0+Math.exp(-x));};
let _sigmoid = function(arr){
	if(typeof arr == `number`) return _func_sigmoid(arr);
	else if(Array.isArray(arr)) return mat.eval(arr, _func_sigmoid);
	else throw "SigmoidException : parameter has not suppoeted type.";
};

/*
ReLU(y) = { x  (x>0)
          { 0  (x<=0)
*/
const _func_relu = (x) => x>0?x:0;
let _relu = function(arr){
	if(typeof arr == `number`) return _func_relu(arr);
	else if(Array.isArray(arr)) return mat.eval(arr, _func_relu);
	else throw "ReLUException : parameter has not suppoeted type.";
};
```


### 2.3 Numerical differentiation
In order to train the neural network, it is necessary to calculate how far the estimated value from with the correct answer or the optimal answer, and to decide the weights should be smaller or larger to reduce the errors. In other words, when we move the weight value a little bit, we will modify the weight to the direction in which the weight decreases. It also shows that the variation of the error is differentiated bt the movement of the weight. It is not easy to obtain the derivative value by analytical method for the entire complex neural network. Since we have already defined neural network model, however, we can obtain the derivative value by using numerical method.

The method of the numerical differentation is finding the slope of the function according to a change of a very small variable. Moreover, [the backpropagation](http://cs231n.github.io/optimization-2/) using the [chain rule of the differentiation](https://en.wikipedia.org/wiki/Chain_rule) is a better way to learn the neural network. It can reduce the number of differentiations for each class of the model. However, since we focus on simple neural network implementation using Javascript, we implemented the simplest way of numerical differentiation without error back propagation. 

```javascript
// [[ math/derivative.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/math/derivative.js

/* Numerical Gradient */
let _numerical_gradient = function(f, x, h=0.0000001) {
	if(typeof f !== `function`) {
		throw "DerivativeException : first parameter is not function";
	} else if(Array.isArray(x)) {
		const _partial_diff = function(arr){				
			let grad = [];
			if(Array.isArray(arr[0])) {
				for(let i = 0 ; i < arr.length ; i++){
					grad.push(_partial_diff(arr[i]));
				}		
			} else {
				for(let i = 0 ; i < arr.length ; i++){
					let temp = arr[i];
					arr[i] = temp+h;
					let dhp = f(arr[i]);

					arr[i] = temp-h;
					let dhn = f(arr[i]);

					arr[i] = temp;
					grad.push((dhp-dhn)/(2.0*h));
				}
			}
			return grad;
		};
		
		return _partial_diff(x);
	} else {
		throw "DerivativeException : second parameter is suitable";
	}
}
```


## 3. Exercise - the XOR problem
As a exersize, we chose [the XOR problem](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b), the simplest form of nonlinear binary classification. The XOR problem is difficult to implement by a linear classification, and it should include at least one hidden layer on the artificial neural network. In this section, we will examine the possibility of constructing a more complex artificial neural network by constructing a model using two hidden layers.


### 3.1 Model Definition
Below is a three-layer neural network that includes estimates of XOR values, error calculations, and training. `W1`, `W2`, and `Wout` are weight values ​​of the first hidden layer, the second hidden layer, and the output layer, and `b1`, `b2`, and `bout` correspond to the bias values ​​of each layer. The parameter allows to set the number of neurons in the hidden layer at the time of model generation, but the number of neurons in each hidden layer is not set in detail. The initial values ​​of the weights and biases are set to any value between -5 and 5 so that they can converge to the solution a little sooner than when initialized to the same value as 0 or 1. Of course, each initial value may have a performance advantage to use [Xavier or He initial values](https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849) depending on the activation function, but we choose a simple method in terms of writing code.

```javascript
// [[ ex02_xor_problem.js ]]
// https://github.com/ivorycirrus/dl4vanillajs-node-example/blob/master/ex02_xor_problem.js

// artificial neural nets with 3 layers
let MultiLayerNet = function(input_size, hidden_size, output_size){
	let thiz = this;

	if(!(thiz.params = storage.read(FILE_PRE_TRAINED))) {
		thiz.params = {
			'W1' : dl.mat.matrix([input_size, hidden_size], x=>(Math.random()*10.0-5.0)),
			'b1' : dl.mat.matrix([1,hidden_size], 0),
			'W2' : dl.mat.matrix([hidden_size, hidden_size], x=>(Math.random()*10.0-5.0)),
			'b2' : dl.mat.matrix([1,hidden_size], 0),
			'Wout' : dl.mat.matrix([hidden_size, output_size], x=>(Math.random()*10.0-5.0)),
			'bout' : dl.mat.matrix([1,output_size], 0)
		};
	}

	// forward process
	thiz.predict = function(x){ /* ... Skip Implimentation ... */ };

	// Loss function
	thiz.loss = function(x, t){ /* ... Skip Implimentation ... */ };

	// Train weights and biases
	thiz.train = function(x, t, batch_size){ /* ... Skip Implimentation ... */ };
};
```

### 3.2 Training Model
The forward propagation is constructed by multiplying the input by the weight and adding the deviation and applying the activation function. All of activation function uses Sigmoid because it is non-linear function and represented between 0 and 1.

```javascript
// [[ ex02_xor_problem.js ]]
// https://github.com/ivorycirrus/dl4vanillajs-node-example/blob/master/ex02_xor_problem.js

// forward process
thiz.predict = function(x){
	// layer 1
	let L1 = dl.mat.mul(x, thiz.params['W1']);
	L1 = dl.mat.add(L1, thiz.params['b1']);
	L1 = dl.actv.sigmoid(L1);
	// layer 2
	let L2 = dl.mat.mul(L1, thiz.params['W2']);
	L2 = dl.mat.add(L2, thiz.params['b2']);
	L2 = dl.actv.sigmoid(L2);
	// output layer
	let Lout = dl.mat.mul(L2, thiz.params['Wout']);
	Lout = dl.mat.add(Lout, thiz.params['bout']);
	Lout = dl.actv.sigmoid(Lout);
	// output
	return Lout;
};
```

[The Cross-Entropy with Logits](https://gombru.github.io/2018/05/23/cross_entropy_loss) function is used to find the error between the result of the net propagation and the correct answer. This function is used to determine the error between the `y` value obtained by the neural network and the correct answer `t` value.

```javascript
// [[ ex02_xor_problem.js ]]
// https://github.com/ivorycirrus/dl4vanillajs-node-example/blob/master/ex02_xor_problem.js

// Loss function
thiz.loss = function(x, t){		
	let y = thiz.predict(x);
	return dl.loss.cross_entropy_with_logits(y, t);
};
```

Finally, we pass the above error function and learning rate on the optimizer to update the weights. The optimizer is designed to find optimal values ​​with a simple slope descent method using numerical derivatives. Initially, it is also considered to applying [the Stochastic Gradient Decent method](https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1) using the mini-batch. However, here are only the general gradient descent method for obtaining the derivative value for each input value is implemented.

```javascript
// [[ ex02_xor_problem.js ]]
// https://github.com/ivorycirrus/dl4vanillajs-node-example/blob/master/ex02_xor_problem.js

// Train weights and biases
thiz.train = function(x, t, batch_size){
	for(let b = 0 ; b < batch_size ; b++){
		let _x = x.slice(b,b+1);
		let _t = t.slice(b,b+1);
		for(i in thiz.params) {
			thiz.params[i] = dl.opt.gradient_decent_optimizer(()=>thiz.loss(_x,_t), thiz.params[i], LEARNING_RATE);
		}
	}
};
```

```javascript
// [[ nn/optimizer.js ]]
// https://github.com/ivorycirrus/dl4vanillajs/blob/master/nn/optimizer.js

/* Gradient Decent Optimizer */
let _gradient_decent_optimizer = function(f, x, lr=0.001){
	if(typeof f !== `function`) {
		throw "OptimizerException : first parameter is not function";
	} else if(!Array.isArray(x)) {
		throw "OptimizerException : second parameter is not array";
	}

	let grad = diff.grad(f, x);
	let trained = mat.add(x, mat.mul(grad, -1.0*lr));

	return trained;
};
```

### 3.3 Evaluation
The following are the learning and evaluation results of the model. As a result of learning as much as Epoch of 2001 with learning rate of `0.01`, final loss is about `0.24` and it can be considered significance four of XOR classified values.

```
$ node ex02_xor_problem.js
==[TRAIN]==
step : 0 loss : 0.9823129621716575
step : 200 loss : 0.5809593623619813
step : 400 loss : 0.5251647745298205
step : 600 loss : 0.4857023698210541
step : 800 loss : 0.4516157963438475
step : 1000 loss : 0.4188023743659243
step : 1200 loss : 0.3851033533690704
step : 1400 loss : 0.34942643976702165
step : 1600 loss : 0.31207226355231443
step : 1800 loss : 0.2750145679970029
step : 2000 loss : 0.24064839885348238
==[TEST]==
Prediction : 0.13 	Correct : 0.00
Prediction : 0.70 	Correct : 1.00
Prediction : 0.89 	Correct : 1.00
Prediction : 0.29 	Correct : 0.00
```

Indeed, learning may not be enough that depends on how the weights are initialized. Below result shows the error is `0.5` or more, and the XOR value with the input value of `[1,0]` is `0.34` Even if the `2001` epochs learned with an arbitrary initial value.

```
==[TRAIN]==
step : 0 loss : 0.9880284306212831
step : 200 loss : 0.6019301518619822
step : 400 loss : 0.5668457641058527
step : 600 loss : 0.5477305276780536
step : 800 loss : 0.5356430221097995
step : 1000 loss : 0.5272774189972985
step : 1200 loss : 0.521101848949871
step : 1400 loss : 0.5163244034156572
step : 1600 loss : 0.512496475979676
step : 1800 loss : 0.5093447391416038
step : 2000 loss : 0.5066926080889506
==[TEST]==
Prediction : 0.36 	Correct : 0.00
Prediction : 0.95 	Correct : 1.00
Prediction : 0.34 	Correct : 1.00
Prediction : 0.35 	Correct : 0.00
```

However, this is a matter of learning speed, and accuracy can be improved through a sufficient number of iterations. The example project contains pre-trained coefficients on [pre_trained/ex02_pretrained_weights.json](https://github.com/ivorycirrus/dl4vanillajs-node-example/blob/master/pre_trained/ex02_pretrained_weights.json). If you put the file name in the `FILE_PRE_TRAINED` variable, you can use the pre-trained initial value instead of arbitary initial value. The pre-trained cofficients contains weight that gives a small error of about `0.005`. Of course, we can see that the error is further reduced by additional trainings.

```
==[TRAIN]==
step : 0 loss : 0.005468257253796859
step : 200 loss : 0.0053710141731233675
step : 400 loss : 0.005277006523766076
step : 600 loss : 0.005186078832492771
step : 800 loss : 0.005098085310905433
step : 1000 loss : 0.00501288911878565
step : 1200 loss : 0.004930361693683351
step : 1400 loss : 0.004850382138454004
step : 1600 loss : 0.004772836663248752
step : 1800 loss : 0.0046976180757801935
step : 2000 loss : 0.004624625311294744
==[TEST]==
Prediction : 0.00 	Correct : 0.00
Prediction : 1.00 	Correct : 1.00
Prediction : 1.00 	Correct : 1.00
Prediction : 0.01 	Correct : 0.00
```

## 4. Conclusion
[Do not reinventing the wheel](https://en.wikipedia.org/wiki/Reinventing_the_wheel) is one of a famous maxim among developers. To put it plainly, there are great tools that have already invented and proven to be performance and reliability. So don't waste your time and not to do the effort to rebuild from scratch. Actually, I also agree with that. This article is about the story such as reinventing the wheel. In the process, however, we could get a closer look at how artificial neural networks estimate the value and how to perform the training neural netrowks to find the optimal value. And it was also an opportunity to think deeply about how to handle array operations in Javascript, especially how to use array built-in functions and accumulators .

Furthermore, if you want to implement Deep Learning with Javascript, I would recommend a tool named [Tensorflow.js](https://js.tensorflow.org/). It is a library developed in the same way as Tensorflow of Python or C ++. It can also use pre-trained data from open-souece repositories, and many excellent examples are also [available in Github](https://github.com/tensorflow/tfjs-examples). I think Javascript can be a good tool for creating AI services and applications in the web environment. And I hope that this post will be a small encouragement to develop a Deep Learning application on web or javascript environments, even if it is not Python.
