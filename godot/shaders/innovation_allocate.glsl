#[compute]
#version 450
layout(local_size_x = 1) in;
// INNOVATION_COMMON
void main() {
    int count = 0, freecount = 0, requested = 0, discarded = 0;
    // Partition valid slots and free slots into one heap/free-stack buffer.
    for (int s = 0; s < p.capacity; s++) {
        if (p.sequence == 0 || meta[2 * s] == INVALID ||
            p.sequence - int(meta[2 * s + 1]) >= p.history - 1) {
            meta[2 * s] = INVALID;
            work[p.capacity - 1 - freecount] = uint(s);
            freecount++;
        } else {
            work[count++] = uint(s);
            work[priorities() + s] = floatBitsToUint(energy(p.sequence - int(meta[2 * s + 1])) *
                                                     dot(delta[s].rgb, delta[s].rgb));
        }
    }
    for (int i = count / 2 - 1; i >= 0; i--)
        sift(i, count);
    requested = count;
    for (int pixel = 0; pixel < pixels(); pixel++) {
        float score = uintBitsToFloat(work[incoming() + pixel]);
        if (score < 0)
            continue;
        requested++;
        uint slot;
        if (count < p.capacity) {
            slot = work[count];
            work[count] = slot;
            count++;
        } else {
            slot = work[0];
            float old = priority(int(slot));
            uint oldpixel = meta[2 * slot], birth = meta[2 * slot + 1];
            discarded++;
            if (score < old ||
                (score == old &&
                 (uint(pixel) < oldpixel || (uint(pixel) == oldpixel && uint(p.sequence) < birth))))
                continue;
        }
        ivec2 q = ivec2(pixel % p.width, pixel / p.width);
        delta[slot] = imageLoad(scene, q) - imageLoad(previous, q);
        meta[2 * slot] = uint(pixel);
        meta[2 * slot + 1] = uint(p.sequence);
        work[priorities() + int(slot)] = floatBitsToUint(score);
        if (count == p.capacity && work[0] == slot)
            sift(0, count);
        else {
            int pos = count - 1;
            while (pos > 0) {
                int parent = (pos - 1) / 2;
                if (!less_slot(work[pos], work[parent]))
                    break;
                uint v = work[pos];
                work[pos] = work[parent];
                work[parent] = v;
                pos = parent;
            }
        }
    }
    for (int s = 0; s < p.capacity; s++)
        if (meta[2 * s] != INVALID) {
            int pixel = int(meta[2 * s]);
            work[nextbase() + s] = work[heads() + pixel];
            work[heads() + pixel] = uint(s);
        }
    work[controls()] = uint(count);
    work[controls() + 1] = uint(requested);
    work[controls() + 2] = uint(discarded);
    work[controls() + 3] = uint(p.sequence);
}
