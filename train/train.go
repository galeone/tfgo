package train


import "github.com/galeone/tfgo/core/example"


// this arr support [][]byte []float32 []int32
// it likes this method in python
// tf.train.BytesList tf.train.Int64List tf.train.FloatList

func StringToFeature(value []string) (exampleFeature *example.Feature) {
	bytesArr :=make([][]byte,0)
	for _,s:=range value{
		bytesArr = append(bytesArr,[]byte(s))
	}
	bytesList := example.BytesList{Value: bytesArr}
	featureBytesList := example.Feature_BytesList{BytesList: &bytesList}
	exampleFeature = &example.Feature{Kind: &featureBytesList}
	return
}

func Float32ToFeature(value []float32) (exampleFeature *example.Feature) {
	floatList := example.FloatList{Value: value}
	featureFloatList := example.Feature_FloatList{FloatList: &floatList}
	exampleFeature = &example.Feature{Kind: &featureFloatList}
	return
}

func Int32ToFeature(value []int64) (exampleFeature *example.Feature) {
	intList := example.Int64List{Value: value}
	featureFloatList := example.Feature_Int64List{Int64List: &intList}
	exampleFeature = &example.Feature{Kind: &featureFloatList}
	return
}