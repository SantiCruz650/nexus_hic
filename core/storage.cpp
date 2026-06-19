#include "storage.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

namespace nexus_hic {

// Función para mapear un archivo binario directamente a memoria (Zero-Copy)
Tensor load_raw_tensor(const std::string& filepath, const std::vector<int64_t>& shape, DataType dtype) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Error: No se pudo abrir el archivo del tensor.");
    }

    // Obtener el tamaño del archivo
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Error: No se pudo obtener el tamaño del archivo.");
    }

    // Mapear el archivo directamente al espacio de memoria virtual
    void* mapped_data = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Error: Falló el mapeo de memoria (mmap).");
    }

    // El descriptor de archivo ya se puede cerrar, el mapa se mantiene vivo
    close(fd);

    Tensor t;
    t.shape = shape;
    t.dtype = dtype;
    t.data_ptr = mapped_data;
    t.byte_size = sb.st_size;
    t.owned_ = true;
    t.compute_strides();

    return t;
}

} // namespace nexus_hic
