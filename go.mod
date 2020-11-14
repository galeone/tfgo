module github.com/galeone/tfgo

go 1.15

require (
	github.com/golang/protobuf v1.4.3 // indirect
	github.com/tensorflow/tensorflow v2.1.0+incompatible
)

replace github.com/tensorflow/tensorflow => github.com/galeone/tensorflow v1.12.2-0.20201110143501-1b6f13331f4d
