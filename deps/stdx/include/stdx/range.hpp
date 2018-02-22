#pragma once

namespace stdx {

///
template<typename T>
class range_iterator {
public:
    using value_type = T;
    using reference = T&;

public:
    explicit range_iterator(T cur) :
        cur(cur)
    { }

    auto operator++ () -> range_iterator& {
        cur += 1;
        return *this;
    }

    auto operator++ (int) -> range_iterator {
        return range_iterator(this->cur + 1)
    }

    auto operator== (range_iterator const& rhs) const {
        return this->cur >= rhs.cur;
    }

    auto operator!= (range_iterator const& rhs) const {
        return this->cur < rhs.cur;
    }

    auto operator*() -> reference { return cur; }

private:
    T cur;
};

///
template<typename T>
class range_t {
public:
    ///
    T _start;
    ///
    T _end;

public:
    using iterator = range_iterator<T>;

public:
    ///
    range_t(T start, T end) :
        _start(start),
        _end(end)
    { }

    ///
    auto begin() const -> iterator {
        return range_iterator<T>(this->_start);
    }

    ///
    auto end() const -> iterator {
        return range_iterator<T>(this->_end);
    }
};

template<typename T>
auto range(T start, T end) -> range_t<T> {
    return range_t<T>(start, end);
}

template<typename T>
auto range(T end) -> range_t<T> {
    return range_t<T>(T(0), end);
}

} // namespace stdx
