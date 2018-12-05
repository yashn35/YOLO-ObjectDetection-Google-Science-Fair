import turicreate as tc

# Load the data
data =  tc.SFrame('/Users/yasnara/Desktop/compost_new_new.sframe')


#Training
# Make a train-test split
train_data, test_data = data.random_split(0.8)
# Create a model

#model = tc.object_detector.create(train_data,feature='image', annotations='annotations', max_iterations=110)
#model.save('temp.model')

model = tc.load_model('/Users/yasnara/Desktop/temp.model')
# Save predictions to an SArray
#print(model.predict(train_data))
#Testing

#model = tc.load_model('/Users/yasnara/Desktop/temp.model')

# Save predictions to an SArray
new_data = "/Users/yasnara/Dekstop/image_new_3.jpeg"

data['image_with_ground_truth'] = \
    tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotations'])
data.explore()

data['image_with_predictions'] = \
    tc.object_detector.util.draw_bounding_boxes(data['image'],model.predict(data['image'], data['annotations']))
data.explore()

#IMPORTANT CODE BELOW
""" 
testdataframe = tc.SFrame()
test_data['image_with_predictions'] = \
    tc.object_detector.util.draw_bounding_boxes(test_data['image'],model.predict(test_data['image']))
test_data.explore()
"""




















#testdataframe.apply(lambda x: x)
#testdataframe.explore()
#testdataframe = tc.object_detector.util.draw_bounding_boxes(data['image'], model.predict(data)[0][0]['coordinates'])
#data['i'] = {'i':tc.object_detector.util.draw_bounding_boxes(data['image'], model.predict(data)[0][0]['coordinates'])}
#data['annotations'] = data['annotations'].apply(lambda x: [x])
#data['image'] = data['image'].apply(lambda x: [x])
#model.predict(test_data)[0][0]['coordinates']

"""
data['image_with_ground_truth'] = \
    tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotations'])

test = tc.SFrame({'image': images})
test['predictions'] = model.predict(test)

test['image_with_predictions'] = \
    tc.object_detector.util.draw_bounding_boxes(test['image'], test['predictions'])
test[['image', 'image_with_predictions']].explore()

"""


#print(predictions)

#print(model.evaluate(test_data))

#model.export_coreml('MyDetector_compost_recycle.mlmodel')




#Other validation code:
"""
import turicreate as tc

data =  tc.SFrame('ig02.sframe')

train, val = data.random_split(0.8)
model = tc.object_detector.create(train)
scores = model.evaluate(val)
print(scores['mean_average_precision'])



"""