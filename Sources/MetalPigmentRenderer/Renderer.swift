import Foundation
import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Pigments (match your Python arrays)
struct Pigment {
    let a: [Float] // length 6
}

let PigmentsTable: [Pigment] = [
    Pigment(a: [0.1, 0.2, 0.8, 0.9, 0.95, 0.99]), // cyan
    Pigment(a: [0.9, 0.7, 0.2, 0.1, 0.05, 0.01]), // magenta
    Pigment(a: [0.2, 0.9, 0.8, 0.2, 0.1, 0.05]), // yellow
    Pigment(a: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   // paper
]

struct Params {
    var width: UInt32
    var height: UInt32
    var bagSize: UInt32
    var nPhotons: UInt32
    var nWavelengths: UInt32
    var nPigments: UInt32
    var seed: UInt32
}

// Canvas: fixed-size bag per pixel, stored as contiguous bytes (UInt8)
final class Canvas {
    let width: Int
    let height: Int
    let bagSize: Int
    var data: [UInt8] // width*height*bagSize

    init(width: Int, height: Int, bagSize: Int) {
        self.width = width
        self.height = height
        self.bagSize = bagSize
        self.data = [UInt8](repeating: 3, count: width*height*bagSize) // default to 'paper' id = 3
    }

    private func bagOffset(x: Int, y: Int) -> Int {
        return (y * width + x) * bagSize
    }

   func applyBrushSquare(pigmentID: UInt8, cx: Int, cy: Int, side: Int, intensity: Float) {
       let baseN = Int(Float(bagSize) * intensity)
       let halfSide = side / 2
       let sigma = Float(side) / 3.0   // adjust spread softness, smaller divisor = harder edges

       for y in max(0, cy - halfSide)...min(height-1, cy + halfSide) {
           for x in max(0, cx - halfSide)...min(width-1, cx + halfSide) {
               let dx = Float(x - cx)
               let dy = Float(y - cy)

               // separable Gaussian falloff
               let w = exp(-(dx*dx) / (2.0 * sigma * sigma)) *
                       exp(-(dy*dy) / (2.0 * sigma * sigma))

               let nAddWeighted = Int(Float(baseN) * w)
               if nAddWeighted == 0 { continue }

               let base = bagOffset(x: x, y: y)
               if nAddWeighted >= bagSize {
                   for i in 0..<bagSize { data[base + i] = pigmentID }
               } else {
                   let keep = bagSize - nAddWeighted
                   for i in 0..<keep {
                       data[base + i] = data[base + bagSize - keep + i]
                   }
                   for i in 0..<nAddWeighted {
                       data[base + keep + i] = pigmentID
                   }
               }
           }
       }
   }
    // mimic appending pigment and capping to bagSize (drop oldest)
    func applyBrush(pigmentID: UInt8, cx: Int, cy: Int, radius: Int, intensity: Float) {
        let baseN = Int(Float(bagSize) * intensity)
        let sigma = Float(radius) / 2.20
        let r2max = Float(radius * radius)

        for y in max(0, cy - radius)...min(height-1, cy + radius) {
            for x in max(0, cx - radius)...min(width-1, cx + radius) {
                let dx = Float(x - cx)
                let dy = Float(y - cy)
                let r2 = dx*dx + dy*dy
                if r2 > r2max { continue }

                // Gaussian falloff
                let w = exp(-r2 / (2.0 * sigma * sigma))
                let nAddWeighted = Int(Float(baseN) * w)
                if nAddWeighted == 0 { continue }

                let base = bagOffset(x: x, y: y)
                if nAddWeighted >= bagSize {
                    for i in 0..<bagSize { data[base + i] = pigmentID }
                } else {
                    let keep = bagSize - nAddWeighted
                    for i in 0..<keep { data[base + i] = data[base + bagSize - keep + i] }
                    for i in 0..<nAddWeighted { data[base + keep + i] = pigmentID }
                }
            }
        }
    }
}

// MARK: - Renderer
final class Renderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "Metal", code: -1, userInfo: [NSLocalizedDescriptionKey: "No Metal device"])
        }
        device = dev
        guard let q = device.makeCommandQueue() else {
            throw NSError(domain: "Metal", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        queue = q

        // Compile the .metal source at runtime (good for SwiftPM)
        let shaderURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("PigmentRenderer.metal")
        let shaderSource = try String(contentsOf: shaderURL, encoding: .utf8)
        let lib = try device.makeLibrary(source: shaderSource, options: nil)
        guard let fn = lib.makeFunction(name: "render_canvas_kernel") else {
            throw NSError(domain: "Metal", code: -3, userInfo: [NSLocalizedDescriptionKey: "Kernel not found"])
        }
        pipeline = try device.makeComputePipelineState(function: fn)
    }

    func run(width: Int, height: Int,
             bagSize: Int, nPhotons: Int,
             seed: UInt32,
             canvas: Canvas,
             outputURL: URL) throws {

        // Build pigment table (flattened floats)
        var pigmentTable: [Float] = []
        for p in PigmentsTable { pigmentTable.append(contentsOf: p.a) }
        let pigmentBuf = device.makeBuffer(bytes: pigmentTable,
                                           length: MemoryLayout<Float>.stride * pigmentTable.count,
                                           options: .storageModeShared)!

        // Params
        var params = Params(width: UInt32(width),
                            height: UInt32(height),
                            bagSize: UInt32(bagSize),
                            nPhotons: UInt32(nPhotons),
                            nWavelengths: 6,
                            nPigments: 4,
                            seed: seed)
        let paramsBuf = device.makeBuffer(bytes: &params,
                                          length: MemoryLayout<Params>.stride,
                                          options: .storageModeShared)!

        // Canvas buffer
        let canvasBuf = device.makeBuffer(bytes: canvas.data,
                                          length: canvas.data.count,
                                          options: .storageModeShared)!

        // Output texture: use rgba32Float so kernel can write float4 directly
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                            width: width,
                                                            height: height,
                                                            mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        guard let outTex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "Metal", code: -5, userInfo: [NSLocalizedDescriptionKey: "Failed to create output texture"])
        }

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw NSError(domain: "Metal", code: -6, userInfo: [NSLocalizedDescriptionKey: "Failed to create command encoder"])
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(paramsBuf, offset: 0, index: 0)
        enc.setBuffer(pigmentBuf, offset: 0, index: 1)
        enc.setBuffer(canvasBuf, offset: 0, index: 2)
        enc.setTexture(outTex, index: 0)

        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tgSize = MTLSizeMake(w, h, 1)
        let grid = MTLSizeMake(width, height, 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        // Read back the float RGBA buffer and convert to UInt8 for PNG
        try saveTextureAsPNG(texture: outTex, url: outputURL)
    }
}

// MARK: - Save rgba32Float texture as PNG (convert to 8-bit)
func saveTextureAsPNG(texture: MTLTexture, url: URL) throws {
    let width = texture.width
    let height = texture.height
    let floatsPerPixel = 4
    let floatCount = width * height * floatsPerPixel
    var floatData = [Float](repeating: 0.0, count: floatCount)

    let bytesPerRow = width * floatsPerPixel * MemoryLayout<Float>.size
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&floatData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

    // Convert float [0,1] to UInt8 [0,255] row-major
    var byteData = [UInt8](repeating: 0, count: width * height * 4)
    for i in 0..<(width * height) {
        let f0 = floatData[i*4 + 0]
        let f1 = floatData[i*4 + 1]
        let f2 = floatData[i*4 + 2]
        let f3 = floatData[i*4 + 3]
        // clamp and convert
        let r = UInt8(max(0.0, min(1.0, f0)) * 255.0)
        let g = UInt8(max(0.0, min(1.0, f1)) * 255.0)
        let b = UInt8(max(0.0, min(1.0, f2)) * 255.0)
        let a = UInt8(max(0.0, min(1.0, f3)) * 255.0)
        byteData[i*4 + 0] = r
        byteData[i*4 + 1] = g
        byteData[i*4 + 2] = b
        byteData[i*4 + 3] = a
    }

    // Create CGImage from raw bytes
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow8 = width * bytesPerPixel
    guard let provider = CGDataProvider(data: Data(byteData) as CFData) else {
        throw NSError(domain: "SavePNG", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create data provider"])
    }

    guard let cg = CGImage(width: width,
                           height: height,
                           bitsPerComponent: 8,
                           bitsPerPixel: 32,
                           bytesPerRow: bytesPerRow8,
                           space: colorSpace,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                           provider: provider,
                           decode: nil,
                           shouldInterpolate: false,
                           intent: .defaultIntent) else {
        throw NSError(domain: "SavePNG", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage"])
    }

    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cg, nil)
    if !CGImageDestinationFinalize(dest) {
        throw NSError(domain: "SavePNG", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize image destination"])
    }
}

// MARK: - Example main (replicates your Python usage)
@main
struct Main {
    static func main() {
        do {
            print("Building for production...")
            let width = 1200, height = 600, bagSize = 500
            let canvas = Canvas(width: width, height: height, bagSize: bagSize)

//   func applyBrushSquare(pigmentID: UInt8, cx: Int, cy: Int, side: Int, intensity: Float)

            // pigment IDs: cyan=0, magenta=1, yellow=2
//            canvas.applyBrush(pigmentID: 0, cx: 400, cy: 300, radius: 300, intensity: 0.8)
//            canvas.applyBrush(pigmentID: 1, cx: 800, cy: 300, radius: 300, intensity: 0.7)
//            canvas.applyBrush(pigmentID: 2, cx: 600, cy: 300, radius: 150, intensity: 0.8)


            canvas.applyBrushSquare(pigmentID: 0, cx: 400, cy: 300, side: 300, intensity: 0.8)
            canvas.applyBrushSquare(pigmentID: 1, cx: 800, cy: 300, side: 300, intensity: 0.7)
            canvas.applyBrushSquare(pigmentID: 2, cx: 600, cy: 300, side: 150, intensity: 0.8)
            let renderer = try Renderer()
            let outURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("paint_strokes.png")
            try renderer.run(width: width,
                             height: height,
                             bagSize: bagSize,
                             nPhotons: 10000,
                             seed: 1337,
                             canvas: canvas,
                             outputURL: outURL)
            print("Wrote \(outURL.path)")
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }
}
