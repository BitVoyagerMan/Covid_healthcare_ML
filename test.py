from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
print(size)



if rank == 0:
   train_dataset_np = np.array(list(train_dataset.as_numpy_iterator()))
   val_dataset_np = np.array(list(val_dataset.as_numpy_iterator()))
   train_dataset_split = np.array_split(train_dataset_np, size)
   val_dataset_split = np.array_split(val_dataset_np, size)
else:
   train_dataset_split = None
   val_dataset_split = None
train_data = comm.scatter(train_dataset_split, root = 0)
val_data = comm.scatter(val_dataset_split, root = 0)
print("wow", type(train_data))
#print(train_data)
train_tf_dataset = tf.data.Dataset.from_tensor_slices(train_data)
val_tf_dataset = tf.data.Dataset.from_tensor_slices(val_data)
#train_tf_dataset = train_tf_dataset.shuffle(buffer_size = 10000)
train_tf_dataset = train_tf_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
val_tf_dataset = val_tf_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
