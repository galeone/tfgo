module github.com/galeone/tfgo

go 1.15

require (
	github.com/golang/protobuf v1.4.3 // indirect
	github.com/tensorflow/tensorflow v2.1.0+incompatible
)

replace github.com/tensorflow/tensorflow => github.com/galeone/tensorflow v2.3.2-0.20201109165538-b577cbc70ba2+incompatible
