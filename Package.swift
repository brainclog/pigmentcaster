// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalPigmentRenderer",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "MetalPigmentRenderer", targets: ["MetalPigmentRenderer"])],
    targets: [
        .executableTarget(
            name: "MetalPigmentRenderer",
            path: "Sources",
            resources: [.process("PigmentRenderer.metal")]
        )
    ]
)

