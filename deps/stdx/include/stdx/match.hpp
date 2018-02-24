
#pragma once

namespace stdx {

template<class... Ts>
struct match_t;

template<>
struct match_t<> {};

template<class T>
struct match_t<T> : T {
    match_t(T t) :
        T { std::move(t) }
    {}

    using T::operator();
};

template<class T0, class... Ts>
struct match_t<T0, Ts...> :
    T0,
    match_t<Ts...>
{
    match_t(T0 t0, Ts... ts) :
        T0 { std::move(t0) },
        match_t<Ts...> { std::move(ts)... }
    {}

    using T0::operator();
    using match_t<Ts...>::operator();
};

template<class... Ts>
auto match(Ts... ts) {
    return match_t<Ts...>(std::move(ts)...);
}

} // namespace stdx
