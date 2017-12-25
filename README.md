201411049 Park Kyung Won
park0kyung0won@gmail.com

# RUN MODE

## 0. CPU ONLY
<pre><code>
$ make cpu 
$ ./cnn_cpu.out 
</code></pre>


## 1. GPU BASIC 
<pre><code>
$ make gpu 
$ ./cnn_gpu.out 
</code></pre>

## 2. GPU TILED 
<pre><code>
$ make gpu_tile 
$ ./cnn_gpu_tile.out 
</code></pre>

## 3. GPU MATRIX MULTIPLICATION 
<pre><code>
$ make gpu_matrix 
$ ./cnn_gpu_matrix.out 
</code></pre>

# UNIT TEST

## 0. Test MNIST data load
<pre><code>
$ make test1
$ ./test_parser1.out 
</code></pre>

## 1. Test weight parameters load 
<pre><code>
$ make test2
$ ./test_parser2.out 
</code></pre>
