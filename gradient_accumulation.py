import numpy as np
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow as tf

class BatchOptimizer(optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True
    def __init__(self, optimizer, maximum_batch_size, desired_batch_size, data_size, model):
        super(BatchOptimizer, self).__init__('batch_optimizer')
        self.model = model        
        if isinstance(self.model, models.Model):
            self.model = [(variable.shape, variable.dtype.name) for variable in self.model.variables]        
        self.optimizer = optimizer
        self._weights = self.optimizer._weights
        self.data_size = data_size
        if self.model[0][1] == 'float16':
            self.dtype = tf.float16
        if self.model[0][1] == 'float32':
            self.dtype = tf.float32
        if self.model[0][1] == 'float64':
            self.dtype = tf.float64
        self.desired_batch_size = desired_batch_size
        self.maximum_batch_size = maximum_batch_size
        self.residual_samples = data_size % desired_batch_size % maximum_batch_size        
        self.data_size_tensor = None
        self.desired_batch_size_tensor = None    
        self.maximum_batch_size_tensor = None   
        self.residual_samples_tensor = None                       
        self.processed_samples_before_update = None 
        self.total_processed_samples = None
        self.accumulated_grads = None                
        self.samples_per_batch = None                        
        if self.desired_batch_size_tensor is None:
            self.desired_batch_size_tensor = tf.constant(self.desired_batch_size, dtype=self.dtype)   
        if self.maximum_batch_size_tensor is None:
            self.maximum_batch_size_tensor = tf.constant(self.maximum_batch_size, dtype=self.dtype)  
        if self.data_size_tensor is None:
            self.data_size_tensor = tf.constant(self.data_size, dtype=self.dtype)  
        if self.residual_samples_tensor is None:
            self.residual_samples_tensor = tf.constant(self.residual_samples, dtype=self.dtype)        
        if self.processed_samples_before_update is None:
            self.processed_samples_before_update = tf.Variable(0.0, trainable=False, dtype=self.dtype)    
        if self.accumulated_grads is None:
            self.accumulated_grads = [tf.Variable(np.zeros(shape, dtype=self.dtype.as_numpy_dtype), shape=shape, trainable=False, dtype=self.dtype) for (shape, _) in self.model]
        if self.samples_per_batch is None:
            self.samples_per_batch = tf.Variable(0.0, trainable=False, dtype=self.dtype)
        if self.total_processed_samples is None:
            self.total_processed_samples = tf.Variable(0.0, trainable=False, dtype=self.dtype)       
                       
    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):          
        
        # Separate gradients and variables because tensorflow graphs do not support zip objects
        grads = []
        variables = []
        for grad, var in grads_and_vars:
            grads.append(grad)
            variables.append(var)
        
        samples_per_batch = self.samples_per_batch
        residual_samples_tensor = self.residual_samples_tensor
        maximum_batch_size_tensor = self.maximum_batch_size_tensor
        
        def update_samples_per_batch_last():
            samples_per_batch.assign(residual_samples_tensor)
            return True
        def update_samples_per_batch():
            samples_per_batch.assign(maximum_batch_size_tensor)
            return False        
        
         # If this is the last batch, update processed samples with residual, else update it with maximum batch size        
        tf.cond(tf.greater(self.total_processed_samples+self.maximum_batch_size_tensor, self.data_size_tensor), 
                update_samples_per_batch_last, update_samples_per_batch)    
        
        self.total_processed_samples.assign_add(self.samples_per_batch)
        self.processed_samples_before_update.assign_add(self.samples_per_batch)
        
        # Accumulate gradients
        for var_index, var in enumerate(variables):                 
            self.accumulated_grads[var_index].assign_add(grads[var_index]*self.samples_per_batch)            

        accumulated_grads = self.accumulated_grads
        processed_samples_before_update = self.processed_samples_before_update
        optimizer = self.optimizer        
        total_processed_samples = self.total_processed_samples     
    
        def return_optimizer_update():
            #Divide the accumulated gradients by the number of samples processed before update               
            for accumulated_grad in accumulated_grads:                
                accumulated_grad.assign(accumulated_grad/processed_samples_before_update)                    
            # Apply gradients with the chosen optimizer
            operation = optimizer.apply_gradients([(accumulated_grads[var_index], var) for var_index, var in enumerate(variables)])                                                      
            #Reset accumulated gradients  
            for accumulated_grad in accumulated_grads:                
                accumulated_grad.assign(np.zeros(accumulated_grad.shape, dtype=accumulated_grad.dtype.as_numpy_dtype))              
            #Reset processed samples before update
            processed_samples_before_update.assign(0.0)
            return operation
        def return_zero():               
            return tf.raw_ops.NoOp()     
        # Check if processed samples before update are greater than the desired batch size
        processed_samples_before_update_greater = tf.greater_equal(self.processed_samples_before_update, self.desired_batch_size_tensor)
        # Check if the total processed samples is greater than the data size
        total_processed_samples_greater = tf.greater_equal(self.total_processed_samples, self.data_size_tensor)
        # If one of the previous conditions applies, update the weights else return no operation
        final_operation = tf.cond(processed_samples_before_update_greater | total_processed_samples_greater,
               return_optimizer_update, return_zero)           
        def reset_total_samples():
            total_processed_samples.assign(0.0)
            return True
        def do_nothing():
            return False
        tf.cond(total_processed_samples_greater, reset_total_samples, do_nothing)
        
        return final_operation
    
    def get_weights(self):
        return self.optimizer.get_weights()
    
    def set_weights(self, weights):        
        self.optimizer.set_weights(weights)
            
    def get_config(self):
        return {'optimizer':optimizers.serialize(self.optimizer), 
                       'maximum_batch_size':self.maximum_batch_size,
                       'desired_batch_size':self.desired_batch_size, 
                       'data_size':self.data_size,
                        'model':self.model
                      } 
    @classmethod
    def from_config(cls, config):
        config['optimizer'] = optimizers.deserialize(config['optimizer'])        
        return cls(**config)
