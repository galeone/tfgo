/*
Copyright 2019 Paolo Galeone. All right reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tfgo

//go:generate protoc -Iframework -Iexample example_parser_configuration.proto --go_out=paths=source_relative:./example
//go:generate protoc -Iframework -Iexample example.proto --go_out=paths=source_relative:./example
//go:generate protoc -Iframework -Iexample feature.proto --go_out=paths=source_relative:./example

//go:generate protoc -Iframework -Iexample resource_handle.proto --go_out=paths=source_relative:./framework
//go:generate protoc -Iframework -Iexample tensor.proto --go_out=paths=source_relative:./framework
//go:generate protoc -Iframework -Iexample tensor_shape.proto --go_out=paths=source_relative:./framework
//go:generate protoc -Iframework -Iexample types.proto --go_out=paths=source_relative:./framework
