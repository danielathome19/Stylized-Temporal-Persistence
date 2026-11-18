#[compute]
#version 450
layout(local_size_x = 1) in;
// SHARED_COMMON
bool less_tile(uint a, uint b) {
    float x = data[prioritybase() + int(a)], y = data[prioritybase() + int(b)];
    return x < y || (x == y && a < b);
}
void sift(int pos) {
    int n = tiles();
    uint value = meta[heapbase() + pos];
    while (2 * pos + 1 < n) {
        int child = 2 * pos + 1;
        if (child + 1 < n && less_tile(meta[heapbase() + child + 1], meta[heapbase() + child]))
            child++;
        if (!less_tile(meta[heapbase() + child], value))
            break;
        meta[heapbase() + pos] = meta[heapbase() + child];
        pos = child;
    }
    meta[heapbase() + pos] = value;
}
void main() {
    int n = tiles(), total = 0;
    if (p.sequence == 0)
        for (int i = 0; i < 16; i++)
            meta[control() + i] = 0;
    for (int t = 0; t < n; t++) {
        total += int(meta[info(t) + 1]);
        meta[heapbase() + t] = uint(t);
    }
    int requested = total, merges = 0;
    for (int i = n / 2 - 1; i >= 0; i--)
        sift(i);
    while (total > p.capacity) {
        int t = int(meta[heapbase()]);
        uint a = meta[info(t) + 2], b = meta[info(t) + 3];
        if (a == INVALID || b == INVALID) {
            meta[control() + 5]++;
            break;
        }
        float cost = data[prioritybase() + t] / image_fraction(t);
        data[ledgerbase() + t] += sqrt(max(cost, 0));
        data[lossbase() + t] += cost;
        for (int h = 0; h < p.history; h++)
            if (meta[mapbase(t) + h] == a)
                meta[mapbase(t) + h] = b;
        meta[alivebase(t) + int(a)] = 0;
        meta[info(t) + 1]--;
        meta[info(t) + 7]++;
        total--;
        merges++;
        propose(t);
        sift(0);
    }
    // Mark all still-live physical versions before any incoming image is written.
    for (int s = 0; s < p.capacity; s++)
        meta[usedbase() + s] = 0;
    for (int t = 0; t < n; t++)
        for (int v = 0; v < int(meta[info(t)]); v++)
            if (meta[alivebase(t) + v] != 0u) {
                uint physical = meta[versionbase(t) + v];
                if (physical < uint(p.capacity))
                    meta[usedbase() + int(physical)] = 1;
            }
    int cursor = 0, stores = 0;
    for (int t = 0; t < n; t++) {
        int incoming = int(meta[info(t) + 5]);
        if (meta[alivebase(t) + incoming] != 0u) {
            while (cursor < p.capacity && meta[usedbase() + cursor] != 0u)
                cursor++;
            if (cursor >= p.capacity) {
                meta[control() + 5]++;
                return;
            }
            meta[versionbase(t) + incoming] = uint(cursor);
            meta[usedbase() + cursor] = 1;
            meta[info(t) + 4] = uint(cursor);
            meta[info(t) + 6]++;
            stores++;
        }
        float totalweight = 0;
        for (int v = 0; v < p.history; v++)
            data[t * p.history + v] = 0;
        for (int age = 0; age < min(p.sequence + 1, p.history); age++) {
            int h = (p.sequence - age) % p.history;
            uint v = meta[mapbase(t) + h];
            float w = weight(age);
            data[t * p.history + int(v)] += w;
            totalweight += w;
        }
        for (int v = 0; v < p.history; v++)
            data[t * p.history + v] /= totalweight;
        // Persistent timestamp indirection contains physical slots between captures.
        for (int h = 0; h < p.history; h++)
            if (meta[mapbase(t) + h] != INVALID)
                meta[mapbase(t) + h] = meta[versionbase(t) + int(meta[mapbase(t) + h])];
    }
    meta[control()] = uint(total);
    meta[control() + 1] = max(meta[control() + 1], uint(total));
    meta[control() + 2] = uint(merges);
    meta[control() + 3] = uint(stores);
    meta[control() + 4] = uint(p.capacity - total);
    meta[control() + 6] = uint(requested);
    meta[control() + 7] = uint(p.sequence);
    meta[control() + 8] = max(meta[control() + 8], uint(requested));
    meta[control() + 9] += uint(merges);
    meta[control() + 10] += uint(stores);
}
